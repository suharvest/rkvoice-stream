"""ctypes runtime for the RK3588 native bridge.

The bridge owns all RKNN ABI structs.  CDLL calls release the Python GIL, so
separate Runtime instances can be used by the runner's branch workers.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import numpy as np
import threading


class _Contract(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32),
                ("input_ndims", ctypes.c_uint32 * 6), ("input_dims", (ctypes.c_uint32 * 4) * 6),
                ("input_fmt", ctypes.c_uint32 * 6), ("input_floats", ctypes.c_uint32 * 6),
                ("output_ndims", ctypes.c_uint32),
                ("output_dims", ctypes.c_uint32 * 4), ("output_fmt", ctypes.c_uint32),
                ("output_floats", ctypes.c_uint32)]


class RK3588CapiRuntime:
    def __init__(self, model_path: str | Path, core_mask: int = 0, library: str | Path | None = None):
        self._lock=threading.RLock(); self._closed=False
        lib = Path(library) if library else Path(__file__).with_name("libkokoro_rk3588_bridge.so")
        self._lib = ctypes.CDLL(str(lib))
        self._lib.kokoro_rk3588_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.kokoro_rk3588_create.restype = ctypes.c_void_p
        self._lib.kokoro_rk3588_query_contract.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Contract)]
        self._lib.kokoro_rk3588_query_contract.restype = ctypes.c_int
        self._lib.kokoro_rk3588_run_float32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        self._lib.kokoro_rk3588_run_float32.restype = ctypes.c_int
        self._lib.kokoro_rk3588_last_error.argtypes = [ctypes.c_void_p]
        self._lib.kokoro_rk3588_last_error.restype = ctypes.c_char_p
        self._lib.kokoro_rk3588_destroy.argtypes = [ctypes.c_void_p]
        self._lib.kokoro_rk3588_destroy.restype = None
        self._handle = self._lib.kokoro_rk3588_create(str(model_path).encode(), int(core_mask))
        if not self._handle:
            raise RuntimeError(f"bridge create failed: {model_path}")
        self.contract = _Contract()
        if self._lib.kokoro_rk3588_query_contract(self._handle, ctypes.byref(self.contract)) != 0:
            self.release(); raise RuntimeError("bridge query contract failed")

    def _error(self):
        return (self._lib.kokoro_rk3588_last_error(self._handle) or b"unknown").decode(errors="replace")

    def inference(self, inputs):
        with self._lock:
            return self._inference(inputs)

    def _inference(self, inputs):
        if self._closed or not self._handle: raise RuntimeError("bridge runtime is closed")
        if len(inputs) != self.contract.n_input:
            raise ValueError(f"expected {self.contract.n_input} inputs, got {len(inputs)}")
        arrays = []
        ptrs = (ctypes.POINTER(ctypes.c_float) * 6)()
        sizes = (ctypes.c_uint32 * 6)()
        for i, value in enumerate(inputs):
            a = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
            if not np.isfinite(a).all(): raise ValueError("input contains non-finite values")
            dims = tuple(int(self.contract.input_dims[i][d]) for d in range(self.contract.input_ndims[i]))
            expected = int(np.prod(dims))
            if a.size != expected:
                raise ValueError(f"input {i} size {a.size} != contract {dims} ({expected})")
            a = a.reshape(dims)
            arrays.append(a)
            ptrs[i] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)); sizes[i] = a.size
        out = np.empty(int(self.contract.output_floats), dtype=np.float32)
        rc = self._lib.kokoro_rk3588_run_float32(self._handle, ptrs, sizes,
                                                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size)
        if rc != 0: raise RuntimeError(f"bridge inference failed: {self._error()}")
        dims = tuple(int(self.contract.output_dims[d]) for d in range(self.contract.output_ndims))
        result = out.reshape(dims)
        # The compiled graphs use NCHW-like [1,C,1,T]. Keep the public
        # runner contract [1,C,T], while accepting a scalar singleton in
        # either of the two middle positions.
        if result.ndim == 4 and result.shape[2] == 1:
            result = result[:, :, 0, :]
        elif result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0, :, :].transpose(0, 2, 1)
        return [result]

    def release(self):
        with getattr(self,"_lock",threading.RLock()):
            if getattr(self, "_handle", None):
                self._lib.kokoro_rk3588_destroy(self._handle); self._handle = None
            self._closed=True

    close = release
