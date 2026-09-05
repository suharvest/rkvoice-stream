"""Receipt-bound persistent Kokoro Conv-only text-to-WAV orchestration.

The trusted manifest authenticates the staged bundle, not independent model
quality. Model files and native libraries are never downloaded by this backend.
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

import numpy as np

from ...engine.tts import TTSBackend
from .kokoro_long32 import CONTEXT_STAGES, PLATFORM_TS, KokoroLong32Runtime, select_profile
from .kokoro_long32_frontend import (
    FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256, LANG_ALIASES, ROUTES, VOICES,
    KokoroLong32Frontend, PhonemeLengthError,
)
from .kokoro_convonly_prefix import OnlinePrefixAdapter
from .kokoro_convonly_tail import PythonConvOnlyTail
from .kokoro_convonly_native import NativeConvOnlyTail
from .kokoro_istft import conv_post_to_waveform

MANIFEST_NAME = "convonly.manifest.json"
MAX_SNAP_RATIO = 0.10
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = 10000
MAX_SEGMENTS = 256


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
    def visit(directory):
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


@dataclass(frozen=True)
class KokoroConvOnlyConfig:
    platform: str
    bundle_root: Path
    manifest_sha256: str
    intra: int | None = None
    inter: int | None = None

    def __post_init__(self):
        if self.platform not in PLATFORM_TS:
            raise ValueError("unsupported platform")
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

    @classmethod
    def from_env(cls):
        def count(name):
            raw = os.environ.get(name)
            if raw is None:
                return None
            if re.fullmatch(r"[0-9]+", raw) is None:
                raise ValueError(f"{name} must be a decimal integer")
            return int(raw)
        root = os.environ.get("KOKORO_CONVONLY_ROOT", "")
        if not root:
            raise ValueError("KOKORO_CONVONLY_ROOT is required")
        return cls(os.environ.get("RK_PLATFORM", "rk3588"), Path(root),
                   os.environ.get("KOKORO_CONVONLY_MANIFEST_SHA256", ""),
                   count("KOKORO_FRONTEND_INTRA_OP_THREADS"),
                   count("KOKORO_FRONTEND_INTER_OP_THREADS"))


def _prefix_parameters(root: Path, manifest: dict, inventory: dict, bundle_root: Path) -> None:
    expected = {}
    for group, blocks in (("rb345", ("rb3", "rb4", "rb5")), ("noise1", ("noise1",))):
        for block in blocks:
            for half in (1, 2):
                for index in range(3):
                    for suffix, count in (("fc-weight", 256 * 128), ("fc-bias", 256), ("alpha", 128)):
                        expected[(group, f"{block}-adain{half}_{index}-{suffix}.bin")] = count
    lineage = manifest.get("lineage")
    rows = lineage.get("parameters") if isinstance(lineage, dict) else None
    if not isinstance(rows, list) or len(rows) != 72:
        raise ValueError("prefix parameter lineage must contain exactly 72 rows")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("invalid prefix parameter lineage row")
        # Builder lineage stores original source paths. Only their canonical
        # basename addresses a parameter inside the staged runtime directory.
        key = (row.get("group"), Path(row["path"]).name)
        if key not in expected or key in seen:
            raise ValueError("prefix parameter set mismatch")
        seen.add(key)
        path = _safe_path(root, f"params/{key[0]}/{key[1]}")
        record = inventory.get(path.relative_to(bundle_root).as_posix())
        if (not record or not _digest(row.get("sha256")) or
                type(row.get("bytes")) is not int or row["bytes"] != expected[key] * 4 or
                (record["sha256"], record["size"]) != (row["sha256"], row["bytes"])):
            raise ValueError(f"prefix parameter identity mismatch: {path.name}")
        values = np.fromfile(path, dtype=np.float32)
        if not np.isfinite(values).all() or (key[1].endswith("-alpha.bin") and np.any(values == 0)):
            raise ValueError(f"invalid prefix parameter values: {path.name}")


def _validate_bundle(config: KokoroConvOnlyConfig) -> dict:
    root = config.bundle_root
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise ValueError(f"symlink in bundle root: {ancestor}")
    if not root.is_dir():
        raise ValueError("bundle root directory missing")
    raw = _safe_path(root, MANIFEST_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.manifest_sha256:
        raise ValueError("trusted manifest SHA mismatch")
    manifest = _json(raw)
    identity = (manifest.get("schema"), manifest.get("platform"), manifest.get("precision"),
                manifest.get("full_source_sha256"), manifest.get("generator_source_sha256"))
    if identity != ("kokoro.convonly.bundle.v1", config.platform, "FP16", FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256):
        raise ValueError("Conv-only manifest platform/schema/source identity mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("bundle files inventory is required")
    inventory = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid bundle file row")
        name = _relative(row.get("path"))
        if name in inventory or name == MANIFEST_NAME:
            raise ValueError("duplicate/self-referential bundle file row")
        if not _digest(row.get("sha256")) or type(row.get("size")) is not int or row["size"] < 0:
            raise ValueError("invalid bundle file SHA/size")
        path = _safe_path(root, name)
        if path.stat().st_size != row["size"] or _sha(path) != row["sha256"]:
            raise ValueError(f"bundle file SHA/size mismatch: {name}")
        inventory[name] = row
    if _files(root) != set(inventory) | {MANIFEST_NAME}:
        raise ValueError("bundle recursive file closure mismatch")
    cpu_generator = manifest.get("cpu_generator")
    if "cpu_generator" in manifest and cpu_generator is None:
        raise ValueError("cpu_generator, when present, must be a relative path")
    if cpu_generator is not None:
        name = _relative(cpu_generator)
        row = inventory.get(name)
        if row is None or row.get("sha256") != GENERATOR_SOURCE_SHA256:
            raise ValueError("cpu_generator must be inventoried with the approved source SHA")
        if row.get("size") != 78971566:
            raise ValueError("cpu_generator size mismatch")
        cpu_generator = _safe_path(root, name)
    frontend = _safe_path(root, manifest.get("frontend"), directory=True)
    prefix = _safe_path(root, manifest.get("prefix"), directory=True)
    fm = _json(_safe_path(frontend, "frontend.manifest.json").read_bytes())
    if fm.get("schema") != "kokoro-v10-long32-frontend-3":
        raise ValueError("Conv-only requires schema-3 frontend")
    pm = _json(_safe_path(prefix, "build-manifest.json").read_bytes())
    # The existing runtime verifies remaining component provenance gates.
    for row in pm.get("rows", []):
        if not isinstance(row, dict) or row.get("stage") not in CONTEXT_STAGES:
            raise ValueError("invalid prefix stage")
        canonical = f"{row['stage']}.fp16.rknn"
        for key in ("rknn", "model"):
            if key in row and row[key] != canonical:
                raise ValueError("noncanonical prefix model basename")
    _prefix_parameters(prefix, pm, inventory, root)
    tail = manifest.get("tail")
    if not isinstance(tail, dict) or tail.get("implementation") != ("native" if config.platform == "rk3576" else "python"):
        raise ValueError("tail implementation/platform mismatch")
    model_root = _safe_path(root, tail.get("model_dir"), directory=True)
    slopes_root = _safe_path(root, tail.get("slopes_dir"), directory=True)
    merge = _safe_path(root, tail.get("merge"))
    library = _safe_path(root, tail.get("library"))
    if config.platform not in merge.name or not merge.name.endswith(".fp16.rknn"):
        raise ValueError("merge must be a target-specific FP16 RKNN")
    models, slope_paths = set(), []
    for branch in range(3):
        for unit in range(3):
            for conv in (1, 2):
                stem = f"branch{branch}_unit{unit}_conv{conv}"
                models.add(_safe_path(model_root, f"{stem}.convonly.{config.platform}.fp16.rknn"))
                slope_paths.append(_safe_path(slopes_root, f"{stem}.npy"))
    if set(model_root.rglob("*.rknn")) != models:
        raise ValueError("tail must contain exactly the 18 target FP16 Conv files")
    if set(slopes_root.rglob("*.npy")) != set(slope_paths):
        raise ValueError("tail must contain exactly 18 slope files")
    slopes = []
    for path in slope_paths:
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32 or value.shape not in ((128,), (1, 128, 1), (1, 128, 1, 1)) or not np.isfinite(value).all():
            raise ValueError(f"slope ABI mismatch: {path.name}")
        slopes.append(value.reshape(128))
    return {"manifest": manifest, "frontend": frontend, "prefix": prefix,
            "model_root": model_root, "merge": merge, "library": library,
            "slopes": np.ascontiguousarray(np.stack(slopes), dtype=np.float32),
            "cpu_generator": cpu_generator}


class ConvOnlyCancelled(RuntimeError):
    def __init__(self, stage):
        super().__init__(f"Conv-only synthesis cancelled after {stage}; submitted work drained")
        self.stage = stage
        self.cancelled = True


class KokoroConvOnlyBackend(TTSBackend):
    # Streaming is sentence-level PCM delivery; model inference remains
    # complete per sentence (there is no intra-sentence model incrementality).
    supports_streaming = True

    def __init__(self, config: KokoroConvOnlyConfig | None = None):
        self.config = KokoroConvOnlyConfig.from_env() if config is None else config
        if not isinstance(self.config, KokoroConvOnlyConfig):
            raise TypeError("config must be KokoroConvOnlyConfig")
        self._lock = threading.RLock()
        self._ready = False
        self._closed = False
        self._close_errors: tuple[str, ...] = ()
        self.frontend = self.prefix_runtime = self.prefix = self.tail = None
        self._bundle = None
        self._cpu_generator_session = None
        self._cpu_generator_path = None

    @property
    def name(self):
        return "kokoro_convonly"

    def is_ready(self):
        # Advisory snapshot only: readiness must not borrow the lifecycle lock
        # or provide admission/resource ownership to a concurrent synthesis.
        return self._ready and not self._closed

    def preload(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            if self._ready:
                return
            try:
                bundle = _validate_bundle(self.config)
                self.frontend = KokoroLong32Frontend(bundle["frontend"],
                    intra_op_num_threads=self.config.intra, inter_op_num_threads=self.config.inter)
                self.prefix_runtime = KokoroLong32Runtime(bundle["prefix"],
                    frontend=self.frontend.render, platform=self.config.platform,
                    context_core_masks={}, route_ts=PLATFORM_TS[self.config.platform], prefix_only=True)
                self.prefix_runtime._manifest()  # Real component gate, before any NPU creation.
                self.frontend.preload_sessions()
                self.prefix_runtime.preload()
                self.prefix = OnlinePrefixAdapter(self.prefix_runtime)
                tail_cls = NativeConvOnlyTail if self.config.platform == "rk3576" else PythonConvOnlyTail
                self.tail = tail_cls(model_root=bundle["model_root"], merge_path=bundle["merge"],
                    slopes=bundle["slopes"], library=bundle["library"],
                    max_n=60 * max(PLATFORM_TS[self.config.platform]) + 1)
                # Detect changes across loading; qualification uses immutable staging.
                _validate_bundle(self.config)
                self._bundle = bundle
                self._cpu_generator_path = bundle.get("cpu_generator")
                self._ready = True
            except BaseException as exc:
                errors = self._release_all()
                self._closed = True
                self._close_errors = tuple(errors)
                if errors:
                    raise RuntimeError(f"preload failed: {exc}; cleanup errors: {'; '.join(errors)}") from exc
                raise

    def _release_all(self):
        self._ready = False
        self.prefix = None
        errors = []
        # atexit can retain the runtime: break its bound frontend reference.
        if self.prefix_runtime is not None:
            self.prefix_runtime.frontend = None
        for name in ("tail", "prefix_runtime", "frontend"):
            component = getattr(self, name)
            if component is None:
                continue
            hook = getattr(component, "close", None) or getattr(component, "cleanup", None)
            try:
                if hook is not None:
                    hook()
            except BaseException as exc:
                errors.append(f"{name}: {exc}")
            else:
                setattr(self, name, None)
        self._bundle = None
        self._cpu_generator_path = None
        try:
            self._release_cpu_generator()
        except BaseException as exc:
            errors.append(f"cpu_generator: {exc}")
        return errors

    def _cpu_generator(self):
        if self._cpu_generator_path is None:
            raise ValueError("duration requires an approved cpu_generator artifact")
        if self._cpu_generator_session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.config.intra
            options.inter_op_num_threads = self.config.inter
            self._cpu_generator_session = ort.InferenceSession(
                str(self._cpu_generator_path), sess_options=options,
                providers=["CPUExecutionProvider"])
        return self._cpu_generator_session

    def _release_cpu_generator(self):
        session = self._cpu_generator_session
        if session is None:
            return
        release = getattr(session, "release", None) or getattr(session, "close", None)
        if release is not None:
            release()
        self._cpu_generator_session = None

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._close_errors = tuple(self._release_all())
            if self._close_errors:
                raise RuntimeError("Conv-only cleanup errors: " + "; ".join(self._close_errors))

    cleanup = close

    @staticmethod
    def _cancel(events, stage):
        if any(event.is_set() for event in events):
            raise ConvOnlyCancelled(stage)

    def _prepare_candidate(self, text, language, speed, cancel):
        self._cancel(cancel, "prepare")
        prepared = self.frontend.prepare(text, language=language, speed=speed,
                                         cancel_check=lambda: self._cancel(cancel, "frontend"))
        self._cancel(cancel, "prepare")
        ids = getattr(prepared, "input_ids", None)
        minimum_d = int(ids.shape[-1]) if ids is not None else 0
        profiles = tuple(t for t in PLATFORM_TS[self.config.platform] if t // 2 >= minimum_d)
        if not profiles:
            raise PhonemeLengthError("no bounded profile can hold the token durations")
        profile = select_profile(prepared.natural_D, max_ratio=MAX_SNAP_RATIO,
                                 platform=self.config.platform, profiles=profiles)
        return prepared, profile

    def _iter_planned_segments(self, text, language, speed, cancel):
        """Lazy lossless partition; the caller holds the lifecycle lock on next()."""
        pending = [(0, len(text))]
        while pending:
            self._cancel(cancel, "planning")
            start, end = pending.pop()
            try:
                prepared, profile = self._prepare_candidate(text[start:end], language, speed, cancel)
            except PhonemeLengthError:
                prepared = profile = None
            if prepared is not None and (not profile["fallback"] or prepared.natural_D <= max(PLATFORM_TS[self.config.platform]) // 2):
                yield start, end, prepared, profile
                continue
            # Keep both pieces nonempty after trimming; trailing/leading spaces
            # remain attached to speech rather than becoming a separate G2P call.
            lo = start + len(text[start:end]) - len(text[start:end].lstrip())
            hi = end - (len(text[start:end]) - len(text[start:end].rstrip()))
            if hi - lo <= 1:
                raise ValueError("single source character exceeds bounded synthesis duration")
            middle = (lo + hi) // 2
            boundaries = [i for i in range(lo + 1, hi)
                          if text[i - 1].isspace() or text[i - 1] in "。！？；，、"]
            split = min(boundaries, key=lambda i: abs(i - middle)) if boundaries else middle
            pending.extend(((split, end), (start, split)))

    @staticmethod
    def _request(text, speaker_id, speed, pitch_shift, kwargs):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
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
        # VoxEdge's PCM-stream API names the same Event protocol cancel_token.
        # Observe both if a caller supplies both; never silently drop either.
        cancel = tuple(kwargs[key] for key in ("cancel_event", "cancel_token") if kwargs.get(key) is not None)
        if any(not callable(getattr(event, "is_set", None)) for event in cancel):
            raise ValueError("cancel_event/cancel_token must provide is_set()")
        return language, route, voice, speed, cancel

    def _render_segment(self, prepared, profile, route, voice, speed, cancel, source_start, source_end):
        with self._lock:
            self._cancel(cancel, "render")
            total = time.perf_counter()
            fallback = bool(profile["fallback"])
            if fallback and self._cpu_generator_path is None:
                raise ValueError(f"duration outside 10% approved profile snap: {profile}")
            # CPUExecutionProvider and RKNN allocate large persistent arenas.
            # Requests are serialized by _lock, so discard the inactive
            # engine before allocating this request's frontend tensors.  Same
            # engine requests still reuse their hot session/profile.
            if fallback:
                self.prefix_runtime.release_profile_contexts()
            else:
                self._release_cpu_generator()
            start = time.perf_counter()
            rendered = self.frontend.render(prepared, target_D=prepared.natural_D if fallback else profile["target_D"])
            frontend_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "render")
            n = rendered["F"] if fallback else profile["F"]
            prefix_ms = 0.0
            boundary = None
            if not fallback:
                start = time.perf_counter()
                boundary = self.prefix.run_prefix(rendered)
                prefix_ms = (time.perf_counter() - start) * 1000
                self._cancel(cancel, "prefix")
                for name, shape in (("j1", (1, 128, n)), ("gamma", (1, 18, 128)), ("beta", (1, 18, 128))):
                    value = np.asarray(getattr(boundary, name))
                    if value.dtype != np.float32 or value.shape != shape or not value.flags.c_contiguous or not np.isfinite(value).all():
                        raise ValueError(f"prefix boundary {name} ABI mismatch")
            start = time.perf_counter()
            if fallback:
                cpu_started = time.perf_counter()
                session = self._cpu_generator()
                self._cancel(cancel, "cpu_generator_load")
                conv_post = session.run(["conv_post"], {"x": rendered["x"], "s": rendered["s"], "har": rendered["har"]})[0]
                tail_info = {"engine": "cpu-ort", "generator": str(self._cpu_generator_path),
                             "cpu_generator_ms": (time.perf_counter() - cpu_started) * 1000}
            else:
                conv_post, tail_info = self.tail.run(boundary.j1, boundary.gamma, boundary.beta)
            tail_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "tail")
            conv_post = np.asarray(conv_post)
            if conv_post.dtype != np.float32 or conv_post.shape != (1, 22, n) or not np.isfinite(conv_post).all():
                raise ValueError("tail output ABI mismatch")
            start = time.perf_counter()
            waveform = conv_post_to_waveform(conv_post)
            if waveform.shape != (1, 5 * (n - 1)) or not np.isfinite(waveform).all():
                raise ValueError("iSTFT output ABI mismatch")
            waveform = waveform[0]
            istft_ms = (time.perf_counter() - start) * 1000
            self._cancel(cancel, "istft")
            start = time.perf_counter()
            peak = float(np.max(np.abs(waveform)))
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
            metadata = {"backend": self.name, "platform": self.config.platform,
                "manifest_sha256": self.config.manifest_sha256,
                "full_source_sha256": FULL_MODEL_SHA256, "generator_source_sha256": GENERATOR_SOURCE_SHA256,
                "route": route, "language": ROUTES[route], "voice": voice, "speed": speed,
                "T": rendered["T"] if fallback else profile["T"], "F": n, "natural_D": prepared.natural_D,
                "target_D": prepared.natural_D if fallback else profile["target_D"], "snap_ratio": 0.0 if fallback else profile["snap_ratio"],
                "route_ts": list(PLATFORM_TS[self.config.platform]), "fallback": fallback,
                "engine": "cpu" if fallback else "npu", "source_start": source_start, "source_end": source_end,
                "fallback_reason": "outside_approved_snap" if fallback else None,
                "cpu_generator_ms": tail_info.get("cpu_generator_ms", 0.0),
                "cancelled": False, "supports_streaming": False, "finite": True, "peak": peak,
                "sample_rate": SAMPLE_RATE, "audio_s": audio_s, "total_ms": total_ms,
                "preload_ms": 0.0, "frontend_ms": frontend_ms, "prefix_ms": prefix_ms,
                "tail_ms": tail_ms, "istft_ms": istft_ms, "wav_ms": wav_ms,
                "frontend_detail": rendered.get("frontend_detail", {}),
                "prefix_detail": {} if boundary is None else dict(boundary.timings), "tail_detail": tail_info,
                "full_rtf": total_ms / 1000 / audio_s, "tail_rtf": tail_ms / 1000 / audio_s}
            return wav_bytes, metadata

    def _iter_audio(self, text, request, *, sentence_mode):
        language, route, voice, speed, cancel = request
        sentences = re.findall(r".+?(?:[。！？；;]+\s*|[.!?]+(?:\s+|$)|$)", text, flags=re.S) if sentence_mode else [text]
        if "".join(sentences) != text:
            raise RuntimeError("source sentence coverage invariant failed")
        source_base = count = 0
        for sentence_index, sentence in enumerate(sentences):
            planner = self._iter_planned_segments(sentence, language, speed, cancel)
            while True:
                with self._lock:
                    started = time.perf_counter()
                    self._cancel(cancel, "request")
                    self.preload()  # Also rejects close() between stream yields.
                    load_ms = (time.perf_counter() - started) * 1000
                    self._cancel(cancel, "preload")
                    plan_started = time.perf_counter()
                    try:
                        start, end, prepared, profile = next(planner)
                    except StopIteration:
                        break
                    plan_ms = (time.perf_counter() - plan_started) * 1000
                    if count >= MAX_SEGMENTS:
                        raise ValueError(f"input exceeds maximum of {MAX_SEGMENTS} segments")
                    wav_bytes, meta = self._render_segment(prepared, profile, route, voice, speed, cancel,
                                                            source_base + start, source_base + end)
                    meta.update(segment_index=count, sentence_index=sentence_index,
                                sentence_text=text[source_base + start:source_base + end],
                                planning_ms=plan_ms, preload_ms=load_ms)
                    meta["frontend_ms"] += plan_ms
                    meta["total_ms"] = (time.perf_counter() - started) * 1000
                    meta["full_rtf"] = meta["total_ms"] / 1000 / meta["audio_s"]
                    count += 1
                # Never retain a thread-bound RLock across a caller-controlled yield.
                yield wav_bytes, meta
            source_base += len(sentence)

    def synthesize(self, text, speaker_id=0, speed=None, pitch_shift=None, **kwargs):
        started = time.perf_counter()
        request = self._request(text, speaker_id, speed, pitch_shift, kwargs)
        with self._lock:
            parts, metas = [], []
            for payload, meta in self._iter_audio(text, request, sentence_mode=False):
                parts.append(payload)
                metas.append(meta)
            self._cancel(request[-1], "wav")
            concat_started = time.perf_counter()
            if len(parts) == 1:
                result = parts[0]
            else:
                output = io.BytesIO()
                with wave.open(output, "wb") as out:
                    out.setnchannels(1); out.setsampwidth(2); out.setframerate(SAMPLE_RATE)
                    for part in parts:
                        with wave.open(io.BytesIO(part), "rb") as wav:
                            out.writeframes(wav.readframes(wav.getnframes()))
                result = output.getvalue()
            concat_ms = (time.perf_counter() - concat_started) * 1000
            engines = {m["engine"] for m in metas}
            metadata = dict(metas[0])
            for key in ("cpu_generator_ms", "preload_ms", "frontend_ms", "prefix_ms", "tail_ms", "istft_ms", "wav_ms", "planning_ms"):
                metadata[key] = sum(m[key] for m in metas)
            metadata["wav_ms"] += concat_ms
            metadata.update(segments=metas, segment_count=len(metas), segmented=len(metas) > 1,
                            engine=next(iter(engines)) if len(engines) == 1 else "mixed",
                            fallback=any(m["fallback"] for m in metas), source_start=0, source_end=len(text),
                            audio_s=sum(m["audio_s"] for m in metas), peak=max(m["peak"] for m in metas))
            if len(metas) > 1:
                # No single bucket represents a concatenation of multiple renders.
                metadata.update(T=None, F=None, natural_D=None, target_D=None, snap_ratio=None)
            self._cancel(request[-1], "wav")
            metadata["total_ms"] = (time.perf_counter() - started) * 1000
            metadata["full_rtf"] = metadata["total_ms"] / 1000 / metadata["audio_s"]
            metadata["tail_rtf"] = metadata["tail_ms"] / 1000 / metadata["audio_s"]
            return result, metadata

    def synthesize_stream(self, text, speaker_id=0, speed=None, pitch_shift=None, **kwargs):
        """Synthesize complete sentences and yield 40 ms float32 chunks.

        The backend lock is held only by each synchronous ``synthesize`` call;
        it is never held across a generator yield.  This is intentionally
        sentence streaming, not incremental model output.
        """
        request = self._request(text, speaker_id, speed, pitch_shift, kwargs)
        for wav_bytes, meta in self._iter_audio(text, request, sentence_mode=True):
            yield from self._stream_wav(wav_bytes, meta, meta["sentence_text"], meta["sentence_index"], request[-1])

    def _stream_wav(self, wav_bytes, meta, sentence, sentence_index, cancel):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                raise ValueError("stream WAV ABI mismatch")
            expected_frames = wav.getnframes(); raw_pcm = wav.readframes(expected_frames)
            if len(raw_pcm) != expected_frames * 2: raise ValueError("stream PCM frame count mismatch")
            # TTSBackend consumers receive normalized float32 samples.  Keep
            # the WAV boundary little-endian PCM16, then convert exactly once.
            pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
        if not pcm.size: raise ValueError("stream WAV is empty")
        audio_s = meta.get("audio_s")
        if audio_s is not None and (not isinstance(audio_s, (int, float)) or not math.isfinite(audio_s)
                                   or abs(audio_s * SAMPLE_RATE - expected_frames) > 0.5):
            raise ValueError("stream metadata duration mismatch")
        stream_meta = dict(meta); stream_meta.update({"streaming_mode": "sentence", "model_incremental": False,
            "supports_streaming": True, "sentence_text": sentence, "sentence_index": sentence_index})
        chunk_size = int(SAMPLE_RATE * 0.04)
        for chunk_index, start in enumerate(range(0, pcm.size, chunk_size)):
            self._cancel(cancel, "chunk")
            chunk_meta = dict(stream_meta); chunk_meta.update({"chunk_index": chunk_index, "segment_complete": start + chunk_size >= pcm.size})
            yield pcm[start:start + chunk_size], chunk_meta

    def get_sample_rate(self):
        return SAMPLE_RATE

    def runtime_info(self):
        with self._lock:
            return {"backend": self.name, "platform": self.config.platform,
                "ready": self._ready and not self._closed, "closed": self._closed,
                "supports_streaming": True, "streaming_mode": "sentence", "model_incremental": False, "manifest_sha256": self.config.manifest_sha256,
                "route_ts": list(PLATFORM_TS[self.config.platform]), "max_snap_ratio": MAX_SNAP_RATIO,
                "frontend_intra": self.config.intra, "frontend_inter": self.config.inter,
                "prefix_core_masks": {key: 0 for key in CONTEXT_STAGES},
                "tail_core_masks": [1, 2, 1] if self.config.platform == "rk3576" else [1, 2, 4],
                "merge_core_mask": 0, "cleanup_errors": list(self._close_errors)}
