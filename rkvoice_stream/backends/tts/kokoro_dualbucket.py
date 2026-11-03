"""Dual-Bucket (T=240, T=480) Kokoro text-to-WAV backend.

Deterministic dual-bucket routing with silence padding and snap duration
conforming to Spec V13.
"""
from __future__ import annotations

import hashlib
import io
import json
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

BUCKET_CONFIGS = {
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


def route_dualbucket(natural_D: int, n_dur: int) -> dict[str, Any]:
    """Deterministic routing state machine for dual bucket (T240, T480).

    Args:
        natural_D: Sum of natural token durations D_nat >= n_dur.
        n_dur: Length of token duration vector N_dur = N_ph + 2.

    Returns:
        Dict with routing decision:
          action: "render" or "split"
          bucket: "T240" or "T480"
          T: 240 or 480
          target_D: 120 or 240
          branch: "padding" or "snap"
          D_out_valid: effective valid frame count
          valid_samples: 600 * D_out_valid
    """
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
    if schema != "kokoro.dualbucket.bundle.v1":
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

    # Models validation (T240, T480) with platform, compiler, and tensor dimension contracts
    models_dict = manifest.get("models")
    if not isinstance(models_dict, dict) or "T240" not in models_dict or "T480" not in models_dict:
        raise ValueError("dual-bucket manifest must specify models for T240 and T480")

    model_paths = {}
    for bucket_key in ("T240", "T480"):
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
    """Orchestrates RKNN / ONNX generator execution across T240 and T480."""

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
                    else:
                        try:
                            from rknnlite.api import RKNNLite
                        except ImportError as err:
                            # Strict fail-closed: do NOT silently fallback to CPU generator unless explicitly allowed
                            if self.allow_cpu_generator and self.cpu_generator is not None:
                                import onnxruntime as ort
                                session = ort.InferenceSession(str(self.cpu_generator), providers=["CPUExecutionProvider"])
                            else:
                                raise RuntimeError("RKNN runtime not available (rknnlite missing or load failed)") from err
                        else:
                            session = RKNNLite(verbose=False)
                            ret = session.load_rknn(str(path))
                            if ret != 0:
                                raise RuntimeError(f"load_rknn failed with code {ret}")
                            ret = session.init_runtime()
                            if ret != 0:
                                raise RuntimeError(f"init_runtime failed with code {ret}")
                    self._sessions[bucket] = session
                except BaseException:
                    if session is not None:
                        self._release_instance(session)
                    raise
            return self._sessions[bucket]

    def run(self, bucket: str, x: np.ndarray, s: np.ndarray, har: np.ndarray) -> np.ndarray:
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

    def preload(self):
        """Warm up and eagerly initialize both T240 and T480 sessions."""
        with self._lock:
            for bucket in ("T240", "T480"):
                self._get_session(bucket)

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
        return "kokoro_convonly"

    def is_ready(self) -> bool:
        return self._ready and not self._closed

    def preload(self) -> None:
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
        decision = route_dualbucket(prepared.natural_D, len(prepared.natural_durations))
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
            else:
                durations = snap_durations(prepared.natural_durations, target_D)
            rendered = self.frontend.render(prepared, target_D=target_D, durations=durations)
            frontend_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "render")

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
                "selected_bucket": bucket,
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
        request = self._request(text, speaker_id, speed, pitch_shift, kwargs)
        yielded = False
        for wav_bytes, meta in self._iter_audio(text, request, sentence_mode=True):
            for chunk in self._stream_wav(wav_bytes, meta, meta["sentence_text"], meta["sentence_index"], request[-1]):
                yielded = True
                yield chunk
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
            return {
                "backend": self.name,
                "platform": self.config.platform,
                "ready": self._ready and not self._closed,
                "closed": self._closed,
                "supports_streaming": True,
                "streaming_mode": "sentence",
                "manifest_sha256": self.config.manifest_sha256,
                "dual_buckets": list(BUCKET_CONFIGS.keys()),
                "cleanup_errors": list(self._close_errors),
            }


# Public alias for parity
KokoroDualBucketBackend = KokoroConvOnlyBackend
