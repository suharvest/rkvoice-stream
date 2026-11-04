"""Dual-Bucket (T=240, T=480) Kokoro text-to-WAV backend.

Deterministic dual-bucket routing with silence padding and snap duration
conforming to Spec V13.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import math
import os
import re
import stat
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence
import unicodedata

import numpy as np

from ...engine.tts import TTSBackend
from .kokoro_convonly import ConvOnlyCancelled
from .kokoro_long32_frontend import (
    FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256, LANG_ALIASES, ROUTES, VOICES,
    KokoroLong32Frontend, PhonemeLengthError, NoPhoneticTokensError, snap_durations,
)
from .kokoro_istft import conv_post_to_waveform

MANIFEST_NAME = "kokoro_dualbucket.manifest.json"
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = 10000
MAX_SEGMENTS = 256
PLATFORMS = ("rk3588", "rk3576")

T_FIX = 50
CTX = 2
STEP = T_FIX - 2 * CTX  # 46 frames
CROSSFADE_SAMPLES = 480  # 20 ms at 24000 Hz
SAMPLES_PER_FRAME = 300

BUCKET_CONFIGS = {
    "T50": {"T": 50, "D_target": 25, "F": 3001},
    "T240": {"T": 240, "D_target": 120, "F": 14401},
    "T480": {"T": 480, "D_target": 240, "F": 28801},
}


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    obj = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(obj, dict):
        raise ValueError("manifest must be an object")
    return obj


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("bundle path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in (".", "..") for p in value.split("/")):
        raise ValueError(f"noncanonical or escaping bundle path: {value}")
    return value


def _safe_path(root: Path, value: object, *, directory: bool = False) -> Path:
    current = root
    for part in _relative(value).split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in bundle path: {current}")
    mode = current.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError(f"wrong bundle path type: {current}")
    return current


def _files(root: Path) -> set[str]:
    found = set()

    def visit(directory: Path):
        for path in directory.iterdir():
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                found.add(path.relative_to(root).as_posix())
            else:
                raise ValueError(f"symlink or special file in bundle: {path}")

    visit(root)
    return found


def _has_pronouncing_tokens(text: str) -> bool:
    """Return True if text contains at least one pronouncing character (letter or number)."""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
    return False


def depop_append(accumulated: np.ndarray, chunk: np.ndarray, crossfade_samples: int = CROSSFADE_SAMPLES) -> np.ndarray:
    """Linearly symmetric crossfade Overlap-Add append to eliminate boundary clicks/pops.

    Args:
        accumulated: Existing 1D float32 waveform buffer.
        chunk: New 1D float32 slice waveform chunk to append.
        crossfade_samples: Number of samples in symmetric linear blend (default 480).

    Returns:
        Concatenated 1D float32 waveform with smooth crossfade.
    """
    if accumulated.size == 0:
        return chunk.astype(np.float32, copy=True)
    overlap = min(crossfade_samples, accumulated.size, chunk.size)
    if overlap <= 0:
        return np.concatenate([accumulated, chunk]).astype(np.float32)
    fade_in = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
    fade_out = 1.0 - fade_in
    blended = accumulated[-overlap:] * fade_out + chunk[:overlap] * fade_in
    return np.concatenate([accumulated[:-overlap], blended, chunk[overlap:]]).astype(np.float32)


def set_big_core_affinity() -> bool:
    """Bind calling thread/process to big CPU cores (cores 4-7 on RK3588/RK3576).

    Checks os.sched_getaffinity(0) if available on Linux.
    If system has >= 8 CPUs (specifically cores 4, 5, 6, 7 are in affinity set),
    bind current thread/process to {4, 5, 6, 7}.
    Gracefully catches and ignores OSError/AttributeError when running in containers
    or non-Linux hosts.

    Returns:
        True if affinity was set, False otherwise.
    """
    try:
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            return False
        current_affinity = os.sched_getaffinity(0)
        target_cores = {4, 5, 6, 7}
        if target_cores.issubset(current_affinity):
            os.sched_setaffinity(0, target_cores)
            return True
    except (OSError, AttributeError, TypeError, ValueError):
        pass
    return False


def apply_silence_padding(natural_durations: np.ndarray | Sequence[int], target_D: int) -> np.ndarray:
    """Pad token durations so speech tokens keep 100% natural duration
    and the trailing EOS token absorbs Delta D = target_D - D_nat.
    """
    d = np.asarray(natural_durations, dtype=np.int64).reshape(-1)
    if d.size == 0 or np.any(d < 1):
        raise ValueError("durations must be non-empty with all values >= 1")
    d_nat = int(d.sum())
    if d_nat > target_D:
        raise ValueError(f"natural duration {d_nat} exceeds target_D {target_D}")
    out = d.copy()
    out[-1] += (target_D - d_nat)
    if np.any(out < 1) or int(out.sum()) != target_D:
        raise RuntimeError("silence padding invariant failed")
    return out


def route_dualbucket(natural_D: int, n_dur: int, has_t50: bool = False) -> dict[str, Any]:
    """Deterministic routing state machine for dual bucket (T240, T480) or windowed T50.

    Args:
        natural_D: Sum of natural token durations D_nat >= n_dur.
        n_dur: Length of token duration vector N_dur = N_ph + 2.
        has_t50: If True, route to T50 windowed generator preserving natural durations.

    Returns:
        Dict with routing decision:
          action: "render" or "split"
          bucket: "T50", "T240" or "T480"
          T: frame dimension (2 * target_D)
          target_D: target frame duration
          branch: "windowed", "padding" or "snap"
          D_out_valid: effective valid frame count
          valid_samples: 600 * D_out_valid
    """
    if has_t50:
        if n_dur > 240:
            return {"action": "split"}
        return {
            "action": "render",
            "bucket": "T50",
            "T": 2 * natural_D,
            "target_D": natural_D,
            "branch": "windowed",
            "D_out_valid": natural_D,
            "valid_samples": 600 * natural_D,
        }

    if n_dur > 240 or natural_D > 282:
        return {"action": "split"}

    # Bucket 1: T=240, D_target=120
    # Requires n_dur <= 120 and natural_D <= 141
    if n_dur <= 120 and natural_D <= 141:
        bucket = "T240"
        T = 240
        target_D = 120
        if natural_D < 105:
            branch = "padding"
            D_out_valid = natural_D
        else:
            branch = "snap"
            D_out_valid = 120
    else:
        # Bucket 2: T=480, D_target=240
        # Entered when 142 <= natural_D <= 282 OR when (n_dur > 120 and natural_D < 209)
        bucket = "T480"
        T = 480
        target_D = 240
        if natural_D < 209:
            branch = "padding"
            D_out_valid = natural_D
        else:
            branch = "snap"
            D_out_valid = 240

    return {
        "action": "render",
        "bucket": bucket,
        "T": T,
        "target_D": target_D,
        "branch": branch,
        "D_out_valid": D_out_valid,
        "valid_samples": 600 * D_out_valid,
    }


@dataclass(frozen=True)
class KokoroDualBucketConfig:
    platform: str
    bundle_root: Path
    manifest_sha256: str
    intra: int | None = None
    inter: int | None = None
    allow_cpu_generator: bool = False

    def __post_init__(self):
        if self.platform not in PLATFORMS:
            raise ValueError(f"unsupported platform: {self.platform!r}")
        if not _digest(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a receipt-bound lowercase SHA256")
        root = Path(self.bundle_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("bundle_root must be absolute without traversal")
        object.__setattr__(self, "bundle_root", root)
        for key, default, minimum in (("intra", 4 if self.platform == "rk3588" else 6, 0), ("inter", 1, 1)):
            value = getattr(self, key)
            value = default if value is None else value
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > 2**31 - 1:
                raise ValueError(f"{key} must be an integer in {minimum}..2147483647")
            object.__setattr__(self, key, value)
        if not isinstance(self.allow_cpu_generator, bool):
            raise TypeError("allow_cpu_generator must be a bool")

    @classmethod
    def from_env(cls):
        def count(name):
            raw = os.environ.get(name)
            if raw is None:
                return None
            if re.fullmatch(r"[0-9]+", raw) is None:
                raise ValueError(f"{name} must be a decimal integer")
            return int(raw)
        root = os.environ.get("KOKORO_DUALBUCKET_ROOT", os.environ.get("KOKORO_CONVONLY_ROOT", ""))
        if not root:
            raise ValueError("KOKORO_DUALBUCKET_ROOT or KOKORO_CONVONLY_ROOT is required")
        sha = os.environ.get("KOKORO_DUALBUCKET_MANIFEST_SHA256", os.environ.get("KOKORO_CONVONLY_MANIFEST_SHA256", ""))
        allow_cpu = os.environ.get("KOKORO_ALLOW_CPU_GENERATOR", "").strip().lower() in ("1", "true", "yes")
        return cls(
            os.environ.get("RK_PLATFORM", "rk3588"),
            Path(root),
            sha,
            count("KOKORO_FRONTEND_INTRA_OP_THREADS"),
            count("KOKORO_FRONTEND_INTER_OP_THREADS"),
            allow_cpu_generator=allow_cpu,
        )


EXPECTED_TENSOR_DIMS = {
    "T50": {
        "inputs": [[1, 512, 50], [1, 128], [1, 22, 3001]],
        "outputs": [[1, 22, 3001]],
    },
    "T240": {
        "inputs": [[1, 512, 240], [1, 128], [1, 22, 14401]],
        "outputs": [[1, 22, 14401]],
    },
    "T480": {
        "inputs": [[1, 512, 480], [1, 128], [1, 22, 28801]],
        "outputs": [[1, 22, 28801]],
    },
}


def _extract_shape(item: Any) -> list[int]:
    """Extract list of ints shape from [dim...] or [name, [dim...]] or {'shape': [dim...]}."""
    if isinstance(item, (list, tuple)):
        if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (list, tuple)):
            return [int(x) for x in item[1]]
        if all(isinstance(x, (int, np.integer)) for x in item):
            return [int(x) for x in item]
    elif isinstance(item, dict) and "shape" in item:
        return [int(x) for x in item["shape"]]
    raise ValueError(f"unrecognized shape descriptor: {item!r}")


def _validate_dualbucket_bundle(config: KokoroDualBucketConfig) -> dict[str, Any]:
    root = config.bundle_root
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise ValueError(f"symlink in bundle root: {ancestor}")
    if not root.is_dir():
        raise ValueError("bundle root directory missing")

    manifest_file = root / MANIFEST_NAME
    if not manifest_file.is_file():
        raise ValueError(f"manifest file missing: {manifest_file}")

    raw = manifest_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.manifest_sha256:
        raise ValueError("trusted manifest SHA mismatch")

    manifest = _json(raw)
    schema = manifest.get("schema")
    if schema not in ("kokoro.dualbucket.bundle.v1", "kokoro.windowed.bundle.v1"):
        raise ValueError(f"unexpected dual-bucket manifest schema: {schema!r}")

    if manifest.get("platform") != config.platform:
        raise ValueError(f"manifest platform mismatch: {manifest.get('platform')} != {config.platform}")
    if manifest.get("precision") != "FP16":
        raise ValueError("manifest precision must be FP16")
    manifest_compiler = manifest.get("compiler", manifest.get("toolkit"))
    if manifest_compiler != "2.3.2":
        raise ValueError(f"manifest compiler/toolkit mismatch: {manifest_compiler!r} != '2.3.2'")
    if manifest.get("full_source_sha256") != FULL_MODEL_SHA256:
        raise ValueError("manifest full_source_sha256 mismatch")
    if manifest.get("generator_source_sha256") != GENERATOR_SOURCE_SHA256:
        raise ValueError("manifest generator_source_sha256 mismatch")

    # Inventory validation
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("bundle files inventory is required")
    inventory = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid bundle file row")
        name = _relative(row.get("path"))
        if name in inventory or name == manifest_file.name:
            raise ValueError("duplicate/self-referential bundle file row")
        if not _digest(row.get("sha256")) or type(row.get("size")) is not int or row["size"] < 0:
            raise ValueError("invalid bundle file SHA/size")
        path = _safe_path(root, name)
        if path.stat().st_size != row["size"] or _sha(path) != row["sha256"]:
            raise ValueError(f"bundle file SHA/size mismatch: {name}")
        inventory[name] = row

    if _files(root) != set(inventory) | {manifest_file.name}:
        raise ValueError("bundle recursive file closure mismatch")

    frontend = _safe_path(root, manifest.get("frontend"), directory=True)
    fm = _json(_safe_path(frontend, "frontend.manifest.json").read_bytes())
    if fm.get("schema") != "kokoro-v10-long32-frontend-3":
        raise ValueError("dual-bucket requires schema-3 frontend")

    # Models validation (T50 windowed, or dual-bucket T240/T480) with platform, compiler, and tensor dimension contracts
    models_dict = manifest.get("models")
    if not isinstance(models_dict, dict) or not models_dict:
        raise ValueError("bundle models inventory is required")
    has_t50 = "T50" in models_dict
    has_dual = "T240" in models_dict and "T480" in models_dict
    if not (has_t50 or has_dual):
        raise ValueError("manifest must specify models for T50 or (T240 and T480)")

    model_paths = {}
    for bucket_key in ("T50", "T240", "T480"):
        if bucket_key not in models_dict:
            continue
        info = models_dict[bucket_key]
        if not isinstance(info, dict):
            raise ValueError(f"invalid model info record for {bucket_key}")

        # Platform check
        b_platform = info.get("platform", manifest.get("platform"))
        if b_platform != config.platform:
            raise ValueError(f"{bucket_key} platform mismatch: {b_platform!r} != {config.platform!r}")

        # Compiler / toolkit check
        b_compiler = info.get("compiler", info.get("toolkit", manifest_compiler))
        if b_compiler != "2.3.2":
            raise ValueError(f"{bucket_key} compiler/toolkit mismatch: {b_compiler!r} != '2.3.2'")

        # Input tensor dimensions check
        raw_inputs = info.get("inputs")
        if not isinstance(raw_inputs, (list, tuple)) or len(raw_inputs) != 3:
            raise ValueError(f"{bucket_key} inputs must specify 3 tensor shapes (x, s, har)")
        input_shapes = [_extract_shape(x) for x in raw_inputs]
        if input_shapes != EXPECTED_TENSOR_DIMS[bucket_key]["inputs"]:
            raise ValueError(
                f"{bucket_key} inputs shape mismatch: expected {EXPECTED_TENSOR_DIMS[bucket_key]['inputs']}, got {input_shapes}"
            )

        # Output tensor dimensions check
        raw_outputs = info.get("outputs")
        if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 1:
            raise ValueError(f"{bucket_key} outputs must specify 1 tensor shape (conv_post)")
        output_shapes = [_extract_shape(x) for x in raw_outputs]
        if output_shapes != EXPECTED_TENSOR_DIMS[bucket_key]["outputs"]:
            raise ValueError(
                f"{bucket_key} outputs shape mismatch: expected {EXPECTED_TENSOR_DIMS[bucket_key]['outputs']}, got {output_shapes}"
            )

        m_name = _relative(info.get("path", info.get("model", f"models/generator_{bucket_key}.rknn")))
        m_path = _safe_path(root, m_name)
        if not m_path.is_file():
            raise ValueError(f"model file missing for {bucket_key}: {m_path}")
        model_paths[bucket_key] = m_path

    cpu_generator = manifest.get("cpu_generator")
    if cpu_generator is not None:
        cpu_generator = _safe_path(root, _relative(cpu_generator))

    return {
        "manifest": manifest,
        "frontend": frontend,
        "models": model_paths,
        "models_info": models_dict,
        "cpu_generator": cpu_generator,
    }


class DualBucketGeneratorRunner:
    """Orchestrates RKNN / ONNX generator execution across T50 windowing and T240/T480."""

    def __init__(
        self,
        models: dict[str, Path],
        models_info: dict,
        platform: str,
        runtime_cls=None,
        cpu_generator: Path | None = None,
        allow_cpu_generator: bool = False,
    ):
        self.models = models
        self.models_info = models_info
        self.platform = platform
        self.runtime_cls = runtime_cls
        self.cpu_generator = cpu_generator
        self.allow_cpu_generator = allow_cpu_generator
        self._sessions = {}
        self._lock = threading.RLock()

    @property
    def has_t50(self) -> bool:
        return "T50" in self.models

    @staticmethod
    def _release_instance(inst: Any) -> None:
        if inst is None:
            return
        hook = getattr(inst, "release", None) or getattr(inst, "close", None)
        if callable(hook):
            try:
                hook()
            except BaseException:
                pass

    def _get_session(self, bucket: str):
        with self._lock:
            if bucket not in self._sessions:
                # Embedded NPU optimization: evict inactive session to avoid IOMMU domain switching
                # contention in RKNPU kernel driver (driver 0.9.8 hangs when multiple active contexts coexist).
                if self.runtime_cls is None and (self.cpu_generator is None or not self.allow_cpu_generator):
                    for old_b in list(self._sessions.keys()):
                        self._release_instance(self._sessions.pop(old_b, None))

                path = self.models[bucket]
                session = None
                try:
                    if self.runtime_cls is not None:
                        session = self.runtime_cls(str(path))
                        init_hook = getattr(session, "init_runtime", None) or getattr(session, "init", None)
                        if callable(init_hook):
                            ret = init_hook()
                            if ret not in (None, 0):
                                raise RuntimeError(f"runtime_cls init failed with code {ret}")
                    elif self.allow_cpu_generator and self.cpu_generator is not None:
                        import onnxruntime as ort
                        session = ort.InferenceSession(str(self.cpu_generator), providers=["CPUExecutionProvider"])
                    else:
                        try:
                            from rknnlite.api import RKNNLite
                            # Restore standard logging level names clobbered by RKNNLite
                            _STD_LEVELS = {"CRITICAL": 50, "FATAL": 50, "ERROR": 40, "WARN": 30, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0}
                            for _name, _lvl in _STD_LEVELS.items():
                                logging.addLevelName(_lvl, _name)
                        except ImportError as err:
                            raise RuntimeError("RKNN runtime not available (rknnlite missing or load failed)") from err
                        else:
                            session = RKNNLite(verbose=False)
                            ret = session.load_rknn(str(path))
                            if ret != 0:
                                raise RuntimeError(f"load_rknn failed with code {ret}")
                            core_mask = getattr(RKNNLite, "NPU_CORE_0_1", 3) if self.platform == "rk3576" else getattr(RKNNLite, "NPU_CORE_AUTO", 0)
                            ret = session.init_runtime(core_mask=core_mask)
                            if ret != 0:
                                raise RuntimeError(f"init_runtime failed with code {ret}")
                    self._sessions[bucket] = session
                except BaseException:
                    if session is not None:
                        self._release_instance(session)
                    raise
            return self._sessions[bucket]

    def run_slice(self, bucket: str, x: np.ndarray, s: np.ndarray, har: np.ndarray) -> np.ndarray:
        with self._lock:
            session = self._get_session(bucket)
            if hasattr(session, "inference"):
                # RKNNLite interface
                outputs = session.inference(inputs=[x, s, har])
                conv_post = outputs[0]
            elif hasattr(session, "run") and not hasattr(session, "get_inputs"):
                # Custom callable or wrapper (e.g. test mock)
                conv_post = session.run(x, s, har)
                if isinstance(conv_post, tuple):
                    conv_post = conv_post[0]
            elif hasattr(session, "get_inputs"):
                # ONNXRuntime session
                outputs = session.run(["conv_post"], {"x": x, "s": s, "har": har})
                conv_post = outputs[0]
            else:
                conv_post = session(x, s, har)
            conv_post = np.asarray(conv_post, dtype=np.float32)
            cfg = BUCKET_CONFIGS[bucket]
            expected_shape = (1, 22, cfg["F"])
            if conv_post.shape != expected_shape or not np.isfinite(conv_post).all():
                raise ValueError(
                    f"generator output ABI mismatch: shape {conv_post.shape} != {expected_shape} "
                    f"or contains non-finite values (all_finite={bool(np.isfinite(conv_post).all())})"
                )
            return conv_post

    def run(self, bucket: str, x: np.ndarray, s: np.ndarray, har: np.ndarray) -> np.ndarray:
        return self.run_slice(bucket, x, s, har)

    def run_windowed(
        self,
        x: np.ndarray,
        s: np.ndarray,
        har: np.ndarray,
        cancel_check: Any = None,
    ) -> tuple[np.ndarray, float, float]:
        """Execute generator using fixed T50 sliding windows and 480-sample Overlap-Add crossfade.

        Returns:
            (waveform, generator_ms, istft_ms)
        """
        x_arr = np.asarray(x, dtype=np.float32)
        s_arr = np.asarray(s, dtype=np.float32)
        har_arr = np.asarray(har, dtype=np.float32)

        if x_arr.ndim != 3 or x_arr.shape[0] != 1 or x_arr.shape[1] != 512:
            raise ValueError(f"x shape mismatch: expected (1, 512, T), got {x_arr.shape}")
        if s_arr.shape != (1, 128):
            raise ValueError(f"s shape mismatch: expected (1, 128), got {s_arr.shape}")
        if har_arr.ndim != 3 or har_arr.shape[0] != 1 or har_arr.shape[1] != 22:
            raise ValueError(f"har shape mismatch: expected (1, 22, F), got {har_arr.shape}")

        T_total = x_arr.shape[2]
        if T_total < 1:
            raise ValueError("x must have at least 1 frame")
        F_total = har_arr.shape[2]
        if F_total < 60 * T_total + 1:
            raise ValueError(f"har length {F_total} insufficient for T={T_total} (requires {60 * T_total + 1})")

        num_windows = 1 if T_total <= T_FIX else int(math.ceil((T_total - T_FIX) / STEP)) + 1
        accumulated = np.empty((0,), dtype=np.float32)
        generator_ms = 0.0
        istft_ms = 0.0

        for k in range(num_windows):
            if cancel_check is not None and callable(cancel_check):
                cancel_check()

            t_start = k * STEP
            t_end = t_start + T_FIX

            if t_end <= T_total:
                x_slice = x_arr[:, :, t_start:t_end]
            else:
                avail = max(0, T_total - t_start)
                x_part = x_arr[:, :, t_start:T_total] if avail > 0 else np.zeros((1, 512, 0), dtype=np.float32)
                n_pad = T_FIX - avail
                x_slice = np.pad(x_part, ((0, 0), (0, 0), (0, n_pad)), mode="constant")

            p_start = t_start * 60
            p_end = p_start + (60 * T_FIX + 1)
            if p_end <= F_total:
                har_slice = har_arr[:, :, p_start:p_end]
            else:
                avail_p = max(0, F_total - p_start)
                har_part = har_arr[:, :, p_start:F_total] if avail_p > 0 else np.zeros((1, 22, 0), dtype=np.float32)
                n_pad_p = (60 * T_FIX + 1) - avail_p
                har_slice = np.pad(har_part, ((0, 0), (0, 0), (0, n_pad_p)), mode="constant")

            t0 = time.perf_counter()
            conv_post_slice = self.run_slice("T50", x_slice, s_arr, har_slice)
            generator_ms += (time.perf_counter() - t0) * 1000

            if cancel_check is not None and callable(cancel_check):
                cancel_check()

            t0 = time.perf_counter()
            w_slice = conv_post_to_waveform(conv_post_slice)[0]
            istft_ms += (time.perf_counter() - t0) * 1000

            if num_windows == 1:
                accumulated = w_slice
            else:
                if k == 0:
                    chunk = w_slice[:14640]
                    accumulated = chunk
                elif k < num_windows - 1:
                    chunk = w_slice[360:14640]
                    accumulated = depop_append(accumulated, chunk, CROSSFADE_SAMPLES)
                else:
                    chunk = w_slice[360:]
                    accumulated = depop_append(accumulated, chunk, CROSSFADE_SAMPLES)

        return accumulated, generator_ms, istft_ms

    def run_windowed_stream(
        self,
        x: np.ndarray,
        s: np.ndarray,
        har: np.ndarray,
        cancel_check: Any = None,
        valid_samples: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Execute generator using fixed T50 sliding windows and yield audio chunks as each window finishes.

        Args:
            x: Acoustic frame tensor (1, 512, T).
            s: Style embedding tensor (1, 128).
            har: Harmonic context tensor (1, 22, F).
            cancel_check: Optional cancellation check callable.
            valid_samples: Optional maximum number of valid samples to emit (e.g. 600 * d_out_valid).

        Yields:
            1D float32 waveform chunks.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        s_arr = np.asarray(s, dtype=np.float32)
        har_arr = np.asarray(har, dtype=np.float32)

        if x_arr.ndim != 3 or x_arr.shape[0] != 1 or x_arr.shape[1] != 512:
            raise ValueError(f"x shape mismatch: expected (1, 512, T), got {x_arr.shape}")
        if s_arr.shape != (1, 128):
            raise ValueError(f"s shape mismatch: expected (1, 128), got {s_arr.shape}")
        if har_arr.ndim != 3 or har_arr.shape[0] != 1 or har_arr.shape[1] != 22:
            raise ValueError(f"har shape mismatch: expected (1, 22, F), got {har_arr.shape}")

        T_total = x_arr.shape[2]
        if T_total < 1:
            raise ValueError("x must have at least 1 frame")
        F_total = har_arr.shape[2]
        if F_total < 60 * T_total + 1:
            raise ValueError(f"har length {F_total} insufficient for T={T_total} (requires {60 * T_total + 1})")

        num_windows = 1 if T_total <= T_FIX else int(math.ceil((T_total - T_FIX) / STEP)) + 1
        target_limit = None if valid_samples is None else max(0, int(valid_samples))
        emitted_samples = 0
        accumulated = np.empty((0,), dtype=np.float32)
        cursor = 0

        def emit(arr: np.ndarray) -> Iterator[np.ndarray]:
            nonlocal emitted_samples
            if arr.size == 0:
                return
            if target_limit is not None:
                remaining = target_limit - emitted_samples
                if remaining <= 0:
                    return
                if arr.size > remaining:
                    arr = arr[:remaining]
            if arr.size > 0:
                emitted_samples += arr.size
                yield arr

        for k in range(num_windows):
            if target_limit is not None and emitted_samples >= target_limit:
                break

            if cancel_check is not None and callable(cancel_check):
                cancel_check()

            t_start = k * STEP
            t_end = t_start + T_FIX

            if t_end <= T_total:
                x_slice = x_arr[:, :, t_start:t_end]
            else:
                avail = max(0, T_total - t_start)
                x_part = x_arr[:, :, t_start:T_total] if avail > 0 else np.zeros((1, 512, 0), dtype=np.float32)
                n_pad = T_FIX - avail
                x_slice = np.pad(x_part, ((0, 0), (0, 0), (0, n_pad)), mode="constant")

            p_start = t_start * 60
            p_end = p_start + (60 * T_FIX + 1)
            if p_end <= F_total:
                har_slice = har_arr[:, :, p_start:p_end]
            else:
                avail_p = max(0, F_total - p_start)
                har_part = har_arr[:, :, p_start:F_total] if avail_p > 0 else np.zeros((1, 22, 0), dtype=np.float32)
                n_pad_p = (60 * T_FIX + 1) - avail_p
                har_slice = np.pad(har_part, ((0, 0), (0, 0), (0, n_pad_p)), mode="constant")

            conv_post_slice = self.run_slice("T50", x_slice, s_arr, har_slice)

            if cancel_check is not None and callable(cancel_check):
                cancel_check()

            w_slice = conv_post_to_waveform(conv_post_slice)[0]

            if num_windows == 1:
                yield from emit(w_slice)
            else:
                # Mirror run_windowed Overlap-Add accumulation exactly:
                if k == 0:
                    chunk = w_slice[:14640]
                    accumulated = chunk
                elif k < num_windows - 1:
                    chunk = w_slice[360:14640]
                    accumulated = depop_append(accumulated, chunk, CROSSFADE_SAMPLES)
                else:
                    chunk = w_slice[360:]
                    accumulated = depop_append(accumulated, chunk, CROSSFADE_SAMPLES)

                # Incremental streaming emit:
                # For intermediate windows (k < num_windows - 1), reserve the final CROSSFADE_SAMPLES
                # because they will be blended with the next window in depop_append().
                # All samples prior to (accumulated.size - CROSSFADE_SAMPLES) are fully mature and immutable.
                if k < num_windows - 1:
                    safe_len = max(0, accumulated.size - CROSSFADE_SAMPLES)
                    if safe_len > cursor:
                        yield from emit(accumulated[cursor:safe_len])
                        cursor = safe_len
                else:
                    # Final window: emit all remaining samples up to the end.
                    if accumulated.size > cursor:
                        yield from emit(accumulated[cursor:])
                        cursor = accumulated.size

    def preload(self):
        """Warm up and eagerly initialize sessions."""
        set_big_core_affinity()
        with self._lock:
            if "T50" in self.models:
                self._get_session("T50")
            for bucket in ("T240", "T480"):
                if bucket in self.models:
                    self._get_session(bucket)
            # If on RKNN hardware, release inactive and keep primary session warm to avoid dual-session IOMMU conflict
            if self.runtime_cls is None and (self.cpu_generator is None or not self.allow_cpu_generator):
                if "T50" in self.models:
                    self._get_session("T50")
                elif "T240" in self.models:
                    self._get_session("T240")

    def close(self):
        with self._lock:
            for s in list(self._sessions.values()):
                self._release_instance(s)
            self._sessions.clear()


class KokoroConvOnlyBackend(TTSBackend):
    """Kokoro Dual-Bucket Standard TTSBackend.

    Conforms to Spec V13. Retains exact backend identity 'kokoro_convonly'
    while supporting deterministic dual-bucket (T240, T480) scheduling.
    """
    supports_streaming = True

    def __init__(self, config: KokoroDualBucketConfig | None = None, runtime_cls=None):
        self.config = KokoroDualBucketConfig.from_env() if config is None else config
        if not isinstance(self.config, KokoroDualBucketConfig):
            raise TypeError("config must be KokoroDualBucketConfig")
        self.runtime_cls = runtime_cls
        self._lock = threading.RLock()
        self._ready = False
        self._closed = False
        self._close_errors: tuple[str, ...] = ()
        self.frontend: KokoroLong32Frontend | None = None
        self.generator: DualBucketGeneratorRunner | None = None
        self._bundle: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        if os.environ.get("TTS_BACKEND") == "kokoro_dualbucket":
            return "kokoro_dualbucket"
        return "kokoro_convonly"

    def is_ready(self) -> bool:
        return self._ready and not self._closed

    def preload(self) -> None:
        set_big_core_affinity()
        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            if self._ready:
                return
            try:
                bundle = _validate_dualbucket_bundle(self.config)
                self.frontend = KokoroLong32Frontend(
                    bundle["frontend"],
                    intra_op_num_threads=self.config.intra,
                    inter_op_num_threads=self.config.inter,
                )
                self.frontend.preload_sessions()
                self.generator = DualBucketGeneratorRunner(
                    bundle["models"],
                    bundle["models_info"],
                    self.config.platform,
                    runtime_cls=self.runtime_cls,
                    cpu_generator=bundle.get("cpu_generator"),
                    allow_cpu_generator=self.config.allow_cpu_generator,
                )
                # Eagerly initialize both NPU sessions before publishing readiness
                self.generator.preload()
                # Re-verify bundle immutability across initialization
                _validate_dualbucket_bundle(self.config)
                self._bundle = bundle
                self._ready = True
            except BaseException as exc:
                errors = self._release_all()
                self._closed = True
                self._close_errors = tuple(errors)
                if errors:
                    raise RuntimeError(f"preload failed: {exc}; cleanup errors: {'; '.join(errors)}") from exc
                raise

    def _release_all(self) -> list[str]:
        self._ready = False
        errors = []
        for name in ("generator", "frontend"):
            comp = getattr(self, name)
            if comp is not None:
                hook = getattr(comp, "close", None) or getattr(comp, "cleanup", None)
                try:
                    if hook is not None:
                        hook()
                except BaseException as exc:
                    errors.append(f"{name}: {exc}")
                else:
                    setattr(self, name, None)
        self._bundle = None
        return errors

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._close_errors = tuple(self._release_all())
            if self._close_errors:
                raise RuntimeError("dual-bucket cleanup errors: " + "; ".join(self._close_errors))

    cleanup = close

    @staticmethod
    def _cancel(events: tuple[Any, ...], stage: str) -> None:
        if any(event.is_set() for event in events):
            raise ConvOnlyCancelled(stage)

    def _prepare_candidate(self, text: str, language: str, speed: float, cancel: tuple[Any, ...]):
        self._cancel(cancel, "prepare")
        prepared = self.frontend.prepare(
            text, language=language, speed=speed,
            cancel_check=lambda: self._cancel(cancel, "frontend"),
        )
        self._cancel(cancel, "prepare")
        if not prepared.input_ids.size or prepared.input_ids.shape[-1] <= 2:
            raise NoPhoneticTokensError("text must contain valid phonetic tokens")
        has_t50 = (self.generator is not None and self.generator.has_t50) or (
            self._bundle is not None and "T50" in self._bundle.get("models", {})
        )
        decision = route_dualbucket(prepared.natural_D, len(prepared.natural_durations), has_t50=has_t50)
        return prepared, decision

    def _iter_planned_segments(self, text: str, language: str, speed: float, cancel: tuple[Any, ...]):
        """Lazy lossless partition along punctuation/whitespace boundaries."""
        pending = [(0, len(text))]
        while pending:
            self._cancel(cancel, "planning")
            start, end = pending.pop()
            candidate_text = text[start:end]
            if not _has_pronouncing_tokens(candidate_text):
                continue
            try:
                prepared, decision = self._prepare_candidate(candidate_text, language, speed, cancel)
            except PhonemeLengthError:
                prepared = decision = None
            except NoPhoneticTokensError:
                continue
            if prepared is not None and decision.get("action") == "render":
                yield start, end, prepared, decision
                continue

            # Need to subdivide candidate text
            lo = start + len(candidate_text) - len(candidate_text.lstrip())
            hi = end - (len(candidate_text) - len(candidate_text.rstrip()))
            if hi - lo <= 1:
                raise ValueError("single source character exceeds bounded synthesis duration")
            middle = (lo + hi) // 2
            boundaries = [i for i in range(lo + 1, hi)
                          if text[i - 1].isspace() or text[i - 1] in "。！？；，、,.!?;:"]
            split = min(boundaries, key=lambda i: abs(i - middle)) if boundaries else middle
            pending.extend(((split, end), (start, split)))

    @staticmethod
    def _request(text: str, speaker_id: int, speed: float | None, pitch_shift: float | None, kwargs: dict):
        if not isinstance(text, str) or not text.strip() or not _has_pronouncing_tokens(text):
            raise NoPhoneticTokensError("text must contain valid phonetic tokens")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"input exceeds maximum length of {MAX_INPUT_CHARS} characters")
        if type(speaker_id) is not int or speaker_id != 0 or pitch_shift not in (None, 0, 0.0):
            raise ValueError("unsupported speaker or pitch shift")
        unknown = set(kwargs) - {"language", "voice", "speaker", "cancel_event", "cancel_token"}
        if unknown:
            raise ValueError("unsupported synthesis options: " + ", ".join(sorted(unknown)))
        language = kwargs.get("language", "en")
        if not isinstance(language, str) or language.strip().lower() not in LANG_ALIASES:
            raise ValueError("unsupported language")
        route = LANG_ALIASES[language.strip().lower()]
        voice = VOICES[route]
        if kwargs.get("voice") not in (None, voice) or kwargs.get("speaker") not in (None, voice):
            raise ValueError(f"voice for route {route} must be {voice}")
        speed = 1.0 if speed is None else speed
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        cancel = tuple(kwargs[k] for k in ("cancel_event", "cancel_token") if kwargs.get(k) is not None)
        if any(not callable(getattr(event, "is_set", None)) for event in cancel):
            raise ValueError("cancel_event/cancel_token must provide is_set()")
        return language, route, voice, speed, cancel

    def _render_segment(self, prepared, decision: dict, route: str, voice: str, speed: float,
                        cancel: tuple[Any, ...], source_start: int, source_end: int):
        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            self._cancel(cancel, "render")
            total = time.perf_counter()
            target_D = decision["target_D"]
            branch = decision["branch"]
            d_out_valid = decision["D_out_valid"]
            bucket = decision["bucket"]

            start = time.perf_counter()
            if branch == "padding":
                durations = apply_silence_padding(prepared.natural_durations, target_D)
            elif branch == "snap":
                durations = snap_durations(prepared.natural_durations, target_D)
            else:
                durations = prepared.natural_durations if target_D == prepared.natural_D else snap_durations(prepared.natural_durations, target_D)
            rendered = self.frontend.render(prepared, target_D=target_D, durations=durations)
            frontend_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "render")

            use_t50 = (self.generator is not None and self.generator.has_t50) or bucket == "T50"
            if use_t50:
                waveform, generator_ms, istft_ms = self.generator.run_windowed(
                    rendered["x"],
                    rendered["s"],
                    rendered["har"],
                    cancel_check=lambda: self._cancel(cancel, "generator"),
                )
                self._cancel(cancel, "istft")
            else:
                start = time.perf_counter()
                conv_post = self.generator.run(bucket, rendered["x"], rendered["s"], rendered["har"])
                generator_ms = (time.perf_counter() - start) * 1000
                self._cancel(cancel, "generator")

                start = time.perf_counter()
                waveform = conv_post_to_waveform(conv_post)[0]
                istft_ms = (time.perf_counter() - start) * 1000
                self._cancel(cancel, "istft")

            # Waveform truncation to valid samples
            valid_samples = 600 * d_out_valid
            waveform = waveform[:valid_samples]

            start = time.perf_counter()
            peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
            pcm = np.clip(np.rint(waveform * 32767.0), -32768, 32767).astype("<i2")
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm.tobytes())
            wav_bytes = output.getvalue()
            wav_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "wav")

            total_ms = (time.perf_counter() - total) * 1000
            audio_s = pcm.size / SAMPLE_RATE
            metadata = {
                "backend": self.name,
                "platform": self.config.platform,
                "manifest_sha256": self.config.manifest_sha256,
                "full_source_sha256": FULL_MODEL_SHA256,
                "generator_source_sha256": GENERATOR_SOURCE_SHA256,
                "route": route,
                "language": ROUTES[route],
                "voice": voice,
                "speed": speed,
                "selected_bucket": "T50" if use_t50 else bucket,
                "branch": branch,
                "T": decision["T"],
                "target_D": target_D,
                "D_out_valid": d_out_valid,
                "natural_D": prepared.natural_D,
                "source_start": source_start,
                "source_end": source_end,
                "audio_s": audio_s,
                "sample_rate": SAMPLE_RATE,
                "peak": peak,
                "total_ms": total_ms,
                "frontend_ms": frontend_ms,
                "generator_ms": generator_ms,
                "istft_ms": istft_ms,
                "wav_ms": wav_ms,
                "cancelled": False,
                "supports_streaming": True,
                "finite": True,
            }
            return wav_bytes, metadata

    def _iter_audio(self, text: str, request: tuple, *, sentence_mode: bool) -> Iterator[tuple[bytes, dict]]:
        language, route, voice, speed, cancel = request
        raw_sentences = re.findall(r".+?(?:[。！？；;]+\s*|[.!?]+(?:\s+|$)|$)", text, flags=re.S) if sentence_mode else [text]
        if "".join(raw_sentences) != text:
            raise RuntimeError("source sentence coverage invariant failed")
        source_base = count = 0
        for sentence_index, sentence in enumerate(raw_sentences):
            if not _has_pronouncing_tokens(sentence):
                source_base += len(sentence)
                continue
            planner = self._iter_planned_segments(sentence, language, speed, cancel)
            while True:
                with self._lock:
                    if self._closed:
                        raise RuntimeError("backend is closed")
                    started = time.perf_counter()
                    self._cancel(cancel, "request")
                    self.preload()
                    load_ms = (time.perf_counter() - started) * 1000
                    self._cancel(cancel, "preload")

                    plan_started = time.perf_counter()
                    try:
                        start, end, prepared, decision = next(planner)
                    except StopIteration:
                        break
                    plan_ms = (time.perf_counter() - plan_started) * 1000
                    if count >= MAX_SEGMENTS:
                        raise ValueError(f"input exceeds maximum of {MAX_SEGMENTS} segments")

                    wav_bytes, meta = self._render_segment(
                        prepared, decision, route, voice, speed, cancel,
                        source_base + start, source_base + end,
                    )
                    meta.update(
                        segment_index=count,
                        sentence_index=sentence_index,
                        sentence_text=text[source_base + start:source_base + end],
                        planning_ms=plan_ms,
                        preload_ms=load_ms,
                    )
                    meta["total_ms"] = (time.perf_counter() - started) * 1000
                    count += 1
                yield wav_bytes, meta
            source_base += len(sentence)

    def synthesize(self, text: str, speaker_id: int = 0, speed: float | None = None,
                   pitch_shift: float | None = None, **kwargs) -> tuple[bytes, dict]:
        started = time.perf_counter()
        request = self._request(text, speaker_id, speed, pitch_shift, kwargs)
        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            parts, metas = [], []
            for payload, meta in self._iter_audio(text, request, sentence_mode=False):
                parts.append(payload)
                metas.append(meta)
            self._cancel(request[-1], "wav")

            if not metas:
                raise NoPhoneticTokensError("text must contain valid phonetic tokens")

            concat_started = time.perf_counter()
            if len(parts) == 1:
                result = parts[0]
            else:
                output = io.BytesIO()
                with wave.open(output, "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(SAMPLE_RATE)
                    for part in parts:
                        with wave.open(io.BytesIO(part), "rb") as wav:
                            out.writeframes(wav.readframes(wav.getnframes()))
                result = output.getvalue()
            concat_ms = (time.perf_counter() - concat_started) * 1000

            metadata = dict(metas[0])
            for key in ("preload_ms", "frontend_ms", "generator_ms", "istft_ms", "wav_ms", "planning_ms"):
                metadata[key] = sum(m.get(key, 0.0) for m in metas)
            metadata["wav_ms"] += concat_ms
            metadata.update(
                segments=metas,
                segment_count=len(metas),
                segmented=len(metas) > 1,
                source_start=0,
                source_end=len(text),
                audio_s=sum(m["audio_s"] for m in metas),
                peak=max(m["peak"] for m in metas),
            )
            if len(metas) > 1:
                metadata.update(T=None, target_D=None, natural_D=None, selected_bucket=None, branch=None)
            self._cancel(request[-1], "wav")
            metadata["total_ms"] = (time.perf_counter() - started) * 1000
            return result, metadata

    def synthesize_stream(self, text: str, speaker_id: int = 0, speed: float | None = None,
                          pitch_shift: float | None = None, **kwargs) -> Iterator[tuple[np.ndarray, dict]]:
        set_big_core_affinity()
        request = self._request(text, speaker_id, speed, pitch_shift, kwargs)
        language, route, voice, speed, cancel = request

        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            self._cancel(cancel, "request")
            self.preload()
            self._cancel(cancel, "preload")

        raw_sentences = re.findall(r".+?(?:[。！？；;]+\s*|[.!?]+(?:\s+|$)|$)", text, flags=re.S)
        if "".join(raw_sentences) != text:
            raise RuntimeError("source sentence coverage invariant failed")

        sentence_plan = []
        offset = 0
        for s_idx, s_raw in enumerate(raw_sentences):
            if _has_pronouncing_tokens(s_raw):
                sentence_plan.append((s_idx, s_raw, offset))
            offset += len(s_raw)

        if not sentence_plan:
            raise NoPhoneticTokensError("text must contain valid phonetic tokens")

        stream_cancel_event = threading.Event()
        effective_cancel = cancel + (stream_cancel_event,)

        def _prepare_and_render_sentence(s_raw: str, s_offset: int, s_idx: int) -> list[dict]:
            set_big_core_affinity()
            self._cancel(effective_cancel, "prepare")
            with self._lock:
                if self._closed:
                    raise RuntimeError("backend is closed")
            segments = []
            for start, end, prepared, decision in self._iter_planned_segments(s_raw, language, speed, effective_cancel):
                self._cancel(effective_cancel, "render")
                target_D = decision["target_D"]
                branch = decision["branch"]
                if branch == "padding":
                    durations = apply_silence_padding(prepared.natural_durations, target_D)
                elif branch == "snap":
                    durations = snap_durations(prepared.natural_durations, target_D)
                else:
                    durations = prepared.natural_durations if target_D == prepared.natural_D else snap_durations(prepared.natural_durations, target_D)

                t0 = time.perf_counter()
                rendered = self.frontend.render(prepared, target_D=target_D, durations=durations)
                frontend_ms = (time.perf_counter() - t0) * 1000

                segments.append({
                    "prepared": prepared,
                    "decision": decision,
                    "rendered": rendered,
                    "frontend_ms": frontend_ms,
                    "source_start": s_offset + start,
                    "source_end": s_offset + end,
                    "sentence_text": s_raw[start:end],
                    "sentence_index": s_idx,
                })
            return segments

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        curr_future: concurrent.futures.Future | None = None
        yielded = False
        segment_counter = 0

        try:
            for i in range(len(sentence_plan)):
                with self._lock:
                    if self._closed:
                        raise RuntimeError("backend is closed")

                # If first sentence, evaluate immediately; otherwise wait for pipelined future
                if curr_future is None:
                    self._cancel(effective_cancel, "prepare")
                    s_cur_idx, s_cur_raw, s_cur_offset = sentence_plan[i]
                    segments = _prepare_and_render_sentence(s_cur_raw, s_cur_offset, s_cur_idx)
                else:
                    try:
                        segments = curr_future.result()
                    finally:
                        curr_future = None

                next_submitted = False
                def _trigger_next_pipeline():
                    nonlocal curr_future, next_submitted
                    if not next_submitted and i + 1 < len(sentence_plan):
                        self._cancel(effective_cancel, "pipeline")
                        s_next_idx, s_next_raw, s_next_offset = sentence_plan[i + 1]
                        curr_future = executor.submit(_prepare_and_render_sentence, s_next_raw, s_next_offset, s_next_idx)
                        next_submitted = True

                for seg in segments:
                    with self._lock:
                        if self._closed:
                            raise RuntimeError("backend is closed")

                    if segment_counter >= MAX_SEGMENTS:
                        raise ValueError(f"input exceeds maximum of {MAX_SEGMENTS} segments")

                    prepared = seg["prepared"]
                    decision = seg["decision"]
                    rendered = seg["rendered"]
                    target_D = decision["target_D"]
                    branch = decision["branch"]
                    d_out_valid = decision["D_out_valid"]
                    bucket = decision["bucket"]
                    valid_samples = 600 * d_out_valid
                    use_t50 = (self.generator is not None and self.generator.has_t50) or bucket == "T50"

                    base_meta = {
                        "backend": self.name,
                        "platform": self.config.platform,
                        "manifest_sha256": self.config.manifest_sha256,
                        "full_source_sha256": FULL_MODEL_SHA256,
                        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
                        "route": route,
                        "language": ROUTES[route],
                        "voice": voice,
                        "speed": speed,
                        "selected_bucket": "T50" if use_t50 else bucket,
                        "branch": branch,
                        "T": decision["T"],
                        "target_D": target_D,
                        "D_out_valid": d_out_valid,
                        "natural_D": prepared.natural_D,
                        "source_start": seg["source_start"],
                        "source_end": seg["source_end"],
                        "audio_s": valid_samples / SAMPLE_RATE,
                        "sample_rate": SAMPLE_RATE,
                        "frontend_ms": seg["frontend_ms"],
                        "streaming_mode": "sentence",
                        "model_incremental": False,
                        "supports_streaming": True,
                        "sentence_text": seg["sentence_text"],
                        "sentence_index": seg["sentence_index"],
                        "segment_index": segment_counter,
                        "cancelled": False,
                        "finite": True,
                    }

                    chunk_size = int(SAMPLE_RATE * 0.04)  # 960 samples = 40ms
                    chunk_idx = 0
                    pending_chunk: np.ndarray | None = None

                    def _yield_pending(data: np.ndarray, complete: bool) -> tuple[np.ndarray, dict]:
                        _trigger_next_pipeline()
                        m = dict(base_meta)
                        m.update({
                            "chunk_index": chunk_idx,
                            "segment_complete": complete,
                            "peak": float(np.max(np.abs(data))) if data.size > 0 else 0.0,
                        })
                        return data, m

                    if use_t50:
                        leftover = np.empty((0,), dtype=np.float32)
                        for w_chunk in self.generator.run_windowed_stream(
                            rendered["x"],
                            rendered["s"],
                            rendered["har"],
                            cancel_check=lambda: self._cancel(cancel, "generator"),
                            valid_samples=valid_samples,
                        ):
                            self._cancel(cancel, "stream")
                            with self._lock:
                                if self._closed:
                                    raise RuntimeError("backend is closed")
                            if w_chunk.size == 0:
                                continue
                            leftover = np.concatenate([leftover, w_chunk])
                            while leftover.size >= chunk_size:
                                self._cancel(cancel, "chunk")
                                pcm_slice = leftover[:chunk_size]
                                leftover = leftover[chunk_size:]
                                if pending_chunk is not None:
                                    out_chunk, out_meta = _yield_pending(pending_chunk, False)
                                    chunk_idx += 1
                                    yielded = True
                                    yield out_chunk, out_meta
                                pending_chunk = pcm_slice

                        if leftover.size > 0:
                            self._cancel(cancel, "chunk")
                            if pending_chunk is not None:
                                out_chunk, out_meta = _yield_pending(pending_chunk, False)
                                chunk_idx += 1
                                yielded = True
                                yield out_chunk, out_meta
                            pending_chunk = leftover

                        if pending_chunk is not None:
                            self._cancel(cancel, "chunk")
                            out_chunk, out_meta = _yield_pending(pending_chunk, True)
                            chunk_idx += 1
                            yielded = True
                            yield out_chunk, out_meta
                            pending_chunk = None

                    else:
                        conv_post = self.generator.run(bucket, rendered["x"], rendered["s"], rendered["har"])
                        self._cancel(cancel, "generator")
                        waveform = conv_post_to_waveform(conv_post)[0][:valid_samples]
                        self._cancel(cancel, "istft")

                        for c_start in range(0, waveform.size, chunk_size):
                            self._cancel(cancel, "chunk")
                            with self._lock:
                                if self._closed:
                                    raise RuntimeError("backend is closed")
                            pcm_slice = waveform[c_start:c_start + chunk_size]
                            if pending_chunk is not None:
                                out_chunk, out_meta = _yield_pending(pending_chunk, False)
                                chunk_idx += 1
                                yielded = True
                                yield out_chunk, out_meta
                            pending_chunk = pcm_slice

                        if pending_chunk is not None:
                            self._cancel(cancel, "chunk")
                            out_chunk, out_meta = _yield_pending(pending_chunk, True)
                            chunk_idx += 1
                            yielded = True
                            yield out_chunk, out_meta
                            pending_chunk = None

                    segment_counter += 1

        finally:
            stream_cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            if curr_future is not None:
                try:
                    curr_future.result(timeout=0.01)
                except Exception:
                    pass

        if not yielded:
            raise NoPhoneticTokensError("text must contain valid phonetic tokens")

    def _stream_wav(self, wav_bytes: bytes, meta: dict, sentence: str, sentence_index: int, cancel: tuple[Any, ...]):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                raise ValueError("stream WAV ABI mismatch")
            expected_frames = wav.getnframes()
            raw_pcm = wav.readframes(expected_frames)
            if len(raw_pcm) != expected_frames * 2:
                raise ValueError("stream PCM frame count mismatch")
            pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
        if not pcm.size:
            raise ValueError("stream WAV is empty")

        stream_meta = dict(meta)
        stream_meta.update({
            "streaming_mode": "sentence",
            "model_incremental": False,
            "supports_streaming": True,
            "sentence_text": sentence,
            "sentence_index": sentence_index,
        })
        chunk_size = int(SAMPLE_RATE * 0.04)
        for chunk_index, start in enumerate(range(0, pcm.size, chunk_size)):
            self._cancel(cancel, "chunk")
            chunk_meta = dict(stream_meta)
            chunk_meta.update({
                "chunk_index": chunk_index,
                "segment_complete": start + chunk_size >= pcm.size,
            })
            yield pcm[start:start + chunk_size], chunk_meta

    def get_sample_rate(self) -> int:
        return SAMPLE_RATE

    def runtime_info(self) -> dict:
        with self._lock:
            models_list = list(self.generator.models.keys()) if self.generator else list(BUCKET_CONFIGS.keys())
            return {
                "backend": self.name,
                "platform": self.config.platform,
                "ready": self._ready and not self._closed,
                "closed": self._closed,
                "supports_streaming": True,
                "streaming_mode": "sentence",
                "manifest_sha256": self.config.manifest_sha256,
                "dual_buckets": models_list,
                "cleanup_errors": list(self._close_errors),
            }


# Public alias for parity
KokoroDualBucketBackend = KokoroConvOnlyBackend
