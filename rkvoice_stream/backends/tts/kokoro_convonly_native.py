"""RK3576 native Conv-only tail adapter.

The shared object is deliberately injected by the caller: importing this module
does not load RKNN, start a service, or mutate the environment.
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from time import monotonic

import numpy as np


class NativeConvOnlyTail:
    """Persistent native tail with strict `[1,C,N]`/`[1,18,C]` contracts."""

    def __init__(self, model_root, merge_path, slopes, library, max_n=38401):
        self._lock = threading.RLock()
        self._handle = None
        self._lib = ctypes.CDLL(str(Path(library)))
        self._lib.kokoro_convonly_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32,
        ]
        self._lib.kokoro_convonly_create.restype = ctypes.c_void_p
        self._lib.kokoro_convonly_run.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char), ctypes.c_uint32,
        ]
        self._lib.kokoro_convonly_run.restype = ctypes.c_int
        self._lib.kokoro_convonly_destroy.argtypes = [ctypes.c_void_p]
        self._lib.kokoro_convonly_destroy.restype = None
        s = np.asarray(slopes)
        if s.dtype != np.float32 or s.shape != (18, 128) or not s.flags.c_contiguous:
            raise ValueError("slopes must be contiguous float32 [18,128]")
        if not np.isfinite(s).all():
            raise ValueError("slopes must be finite")
        if isinstance(max_n, bool) or not isinstance(max_n, int) or not 1 <= max_n <= 38401:
            raise ValueError("max_n must be an integer in 1..38401")
        self.max_n = max_n
        self._slopes = s
        self._err = ctypes.create_string_buffer(512)
        handle = self._lib.kokoro_convonly_create(
            os_path(model_root), os_path(merge_path),
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), s.size, max_n,
            self._err, len(self._err),
        )
        if not handle:
            raise RuntimeError(self._message("native create failed"))
        self._handle = handle

    @staticmethod
    def _array(value, shape, name):
        a = np.asarray(value)
        if a.dtype != np.float32 or a.shape != shape or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous float32 {shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} must be finite")
        return a

    def _message(self, fallback):
        raw = bytes(self._err).split(b"\0", 1)[0].decode(errors="replace")
        return raw or fallback

    def run(self, j1, gamma, beta):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("native tail is closed")
            raw_j = np.asarray(j1)
            if raw_j.ndim != 3:
                raise ValueError("j1 must be contiguous float32 [1,128,N]")
            j = self._array(j1, (1, 128, raw_j.shape[-1]), "j1")
            n = j.shape[-1]
            if n <= 0 or n > self.max_n:
                raise ValueError(f"j1 length {n} outside 1..{self.max_n}")
            g = self._array(gamma, (1, 18, 128), "gamma")
            b = self._array(beta, (1, 18, 128), "beta")
            out = np.empty((1, 22, n), dtype=np.float32)
            started = monotonic()
            rc = self._lib.kokoro_convonly_run(
                self._handle, n,
                j.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), j.size,
                g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), g.size,
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), b.size,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size,
                self._err, len(self._err),
            )
            elapsed = monotonic() - started
            if rc != 0:
                raise RuntimeError(self._message("native tail run failed"))
            if not np.isfinite(out).all():
                raise RuntimeError("native tail returned nonfinite output")
            return out, {"elapsed_s": elapsed, "tail_ms_including_merge": elapsed * 1000,
                         "n": n, "masks": [1, 2, 1],
                         "schedule": "all", "precision": "FP16", "backend": "rk3576-native"}

    def close(self):
        with self._lock:
            handle, self._handle = self._handle, None
            if handle:
                self._lib.kokoro_convonly_destroy(handle)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def os_path(value):
    return str(Path(value)).encode()
