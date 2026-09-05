"""Official Kokoro v1.0 multilingual frontend for the long32 RKNN path.

This module owns text -> phoneme ids -> duration scheduling -> generator
boundary tensors.  It intentionally does not select a T profile or perform
iSTFT; those decisions belong to the production generator runtime.
"""
from __future__ import annotations

import importlib
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

ROUTES = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-BR", "z": "zh"}
LANG_ALIASES = {**{k: k for k in ROUTES}, **{v.lower(): k for k, v in ROUTES.items()}, "en": "a", "pt": "p", "cn": "z", "zh-cn": "z"}
VOICES = {"a": "af_heart", "b": "bf_emma", "e": "ef_dora", "f": "ff_siwis", "h": "hf_alpha", "i": "if_sara", "j": "jf_alpha", "p": "pf_dora", "z": "zf_xiaobei"}
VOICE_SHAPE = (510, 1, 256)
FULL_MODEL_SHA256 = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
GENERATOR_SOURCE_SHA256 = "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c"


class PhonemeLengthError(ValueError):
    """Input phonemes exceed the fixed frontend ABI limit."""
OFFICIAL_SOURCE_SHA256 = FULL_MODEL_SHA256
STEM_TEXT = "/encoder/predictor/text_encoder/Concat_4_output_0"
STEM_STYLE = "/encoder/Slice_output_0"


def snap_durations(durations: np.ndarray, target_D: int) -> np.ndarray:
    d = np.asarray(durations, dtype=np.int64).reshape(-1)
    if d.size == 0 or np.any(d < 1) or target_D < d.size:
        raise ValueError("invalid duration target")
    if int(d.sum()) == target_D:
        return d.copy()
    excess = d - 1
    total = int(excess.sum())
    if total <= 0:
        out = np.ones_like(d)
        q, r = divmod(target_D - d.size, d.size)
        out += q
        if r:
            out[:r] += 1
        return out
    raw = excess.astype(np.float64) * (target_D - d.size) / total
    out = 1 + np.floor(raw).astype(np.int64)
    for i in np.argsort(-(raw - np.floor(raw)), kind="stable")[: target_D - int(out.sum())]:
        out[i] += 1
    if np.any(out < 1) or int(out.sum()) != target_D:
        raise RuntimeError("duration snap invariant failed")
    return out


@dataclass(frozen=True)
class Prepared:
    route: str
    language: str
    text: str
    input_ids: np.ndarray
    style: np.ndarray
    speed: np.ndarray
    natural_durations: np.ndarray
    natural_D: int
    stem_text: np.ndarray | None = None
    stem_style: np.ndarray | None = None
    timings: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own immutable request tensors and keep duration metadata coherent."""
        if self.route not in ROUTES or self.language != ROUTES[self.route]:
            raise ValueError("Prepared route/language mismatch")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Prepared text must be non-empty")

        def freeze(value: Any, dtype: Any) -> np.ndarray:
            contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            # bytes is an immutable backing store, so callers cannot undo the
            # read-only flag with ndarray.setflags(write=True).
            return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)

        arrays = {
            "input_ids": (self.input_ids, np.int64),
            "style": (self.style, np.float32),
            "speed": (self.speed, np.float32),
            "natural_durations": (self.natural_durations, np.int64),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, freeze(value, dtype))
        for name in ("stem_text", "stem_style"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, freeze(value, np.float32))
        if (self.stem_text is None) != (self.stem_style is None):
            raise ValueError("Prepared stem tensors must be both present or both absent")
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("Prepared input_ids must have shape (1,N)")
        if self.style.shape != (1, 256) or not np.isfinite(self.style).all():
            raise ValueError("Prepared style must be finite float32 (1,256)")
        if self.speed.shape != (1,) or not np.isfinite(self.speed).all() or float(self.speed[0]) <= 0:
            raise ValueError("Prepared speed must be finite positive float32 (1,)")
        durations = self.natural_durations
        if durations.ndim != 1 or durations.size != self.input_ids.shape[1] or durations.size == 0 or np.any(durations < 1):
            raise ValueError("Prepared natural durations must be positive and match input_ids")
        natural_d = int(durations.sum())
        if isinstance(self.natural_D, bool) or not isinstance(self.natural_D, (int, np.integer)) or int(self.natural_D) != natural_d:
            raise ValueError("Prepared natural_D does not match natural durations")
        object.__setattr__(self, "natural_D", natural_d)
        if self.stem_text is not None:
            if self.stem_text.shape != (1, durations.size, 640) or self.stem_style.shape != (1, 128):
                raise ValueError("Prepared stem tensor shape mismatch")
            if not np.isfinite(self.stem_text).all() or not np.isfinite(self.stem_style).all():
                raise ValueError("Prepared stem tensors must be finite")
        timings: dict[str, float] = {}
        for key, value in dict(self.timings or {}).items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError("Prepared timings must map strings to finite non-negative numbers")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0:
                raise ValueError("Prepared timings must map strings to finite non-negative numbers")
            timings[key] = numeric
        object.__setattr__(self, "timings", MappingProxyType(timings))


class KokoroLong32Frontend:
    def __init__(self, model_dir: str | Path, *, probe_session: Any = None, target_session: Any = None, stem_session: Any = None, duration_session: Any = None, target_tail_session: Any = None, g2p_factory: Callable[[str], Callable[[str], str]] | None = None, verify_manifest: bool = True, intra_op_num_threads: int | None = None, inter_op_num_threads: int | None = None):
        self.model_dir = Path(model_dir)
        self._init_lock = threading.Lock()
        self._g2p_lock = threading.Lock()
        self._g2p_call_locks: dict[str, threading.Lock] = {}
        self._g2p_cache: dict[str, Callable[[str], str]] = {}
        self._voice_cache: dict[str, np.ndarray] = {}
        self._session_options = self._session_options_factory(intra_op_num_threads, inter_op_num_threads)
        legacy_any = any(x is not None for x in (probe_session, target_session))
        split_any = any(x is not None for x in (stem_session, duration_session, target_tail_session))
        split_all = all(x is not None for x in (stem_session, duration_session, target_tail_session))
        if (legacy_any and split_any) or ((probe_session is None) != (target_session is None)) or (split_any and not split_all):
            raise ValueError("ambiguous or partial frontend session injection")
        if verify_manifest:
            self._verify_manifest()
        else:
            self._manifest_schema = "kokoro-v10-long32-frontend-3" if split_any else "kokoro-v10-long32-frontend-2"
        if legacy_any and self._manifest_schema != "kokoro-v10-long32-frontend-2":
            raise ValueError("legacy sessions require a v2 frontend manifest")
        if split_any and self._manifest_schema != "kokoro-v10-long32-frontend-3":
            raise ValueError("split sessions require a v3 frontend manifest")
        self.vocab = self._load_vocab()
        self.g2p_factory = g2p_factory or self._official_g2p
        self._g2p_factory_seen = self.g2p_factory
        self.stem = stem_session
        self._legacy_sessions = legacy_any
        self.duration_head = duration_session or probe_session
        self.target_tail = target_tail_session or target_session
        # Compatibility aliases used by the pre-split runtime.
        self.probe = self.duration_head
        self.target = self.target_tail
        self._duration_text_name = STEM_TEXT
        self._duration_style_name = STEM_STYLE
        self._target_text_name = STEM_TEXT
        self._target_stem_style_name = STEM_STYLE
        self._split_ready = False
        if split_any and self._manifest_schema == "kokoro-v10-long32-frontend-3":
            names = self._resolve_split_input_names(stem_session, duration_session, target_tail_session)
            self._publish_split_bundle(stem_session, duration_session, target_tail_session, names)

    @staticmethod
    def _thread_count(value: int | None, env_name: str) -> int | None:
        if value is None:
            raw = os.environ.get(env_name)
            if raw is None: return None
            if not raw or raw.strip() != raw or not raw.isdecimal():
                raise ValueError(f"{env_name} must be a non-negative decimal integer")
            value = int(raw, 10)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{env_name} must be a non-negative integer")
        return value

    @classmethod
    def _session_options_factory(cls, intra: int | None, inter: int | None):
        intra = cls._thread_count(intra, "KOKORO_FRONTEND_INTRA_OP_THREADS")
        inter = cls._thread_count(inter, "KOKORO_FRONTEND_INTER_OP_THREADS")
        if intra is None and inter is None: return None
        import onnxruntime as ort
        options = ort.SessionOptions()
        if intra is not None: options.intra_op_num_threads = intra
        if inter is not None: options.inter_op_num_threads = inter
        return options

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_manifest(self) -> None:
        """Validate the immutable frontend ABI before loading any model file."""
        if not self.model_dir.is_absolute() or self.model_dir.is_symlink() or not self.model_dir.is_dir():
            raise ValueError("frontend root must be an existing absolute non-symlink directory")
        manifest_path = self.model_dir / "frontend.manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("frontend.manifest.json is required")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid frontend manifest: {exc}") from exc
        schema = manifest.get("schema")
        if schema not in {"kokoro-v10-long32-frontend-2", "kokoro-v10-long32-frontend-3"} or manifest.get("status") != "PASS":
            raise ValueError("frontend manifest schema/status mismatch")
        if manifest.get("official") is not True or manifest.get("routes") != ROUTES or manifest.get("voices") != VOICES:
            raise ValueError("frontend manifest routes/voices mismatch")
        if (manifest.get("full_source_sha256"), manifest.get("generator_source_sha256")) != (FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256):
            raise ValueError("frontend manifest dual source SHA mismatch")
        export_sha = manifest.get("generator_export_manifest_sha256")
        if not isinstance(export_sha, str) or len(export_sha) != 64:
            raise ValueError("frontend manifest generator export manifest SHA missing")
        export_meta = manifest.get("generator_export_manifest")
        if not isinstance(export_meta, dict) or export_meta.get("schema") != "kokoro-v10-generator-export-1" or export_meta.get("status") != "PASS" or export_meta.get("sha256") != export_sha:
            raise ValueError("frontend manifest generator export chain mismatch")
        export_path = self.model_dir / str(export_meta.get("path", ""))
        if export_path.name != export_meta.get("path") or export_path.is_symlink() or not export_path.is_file() or self._sha256(export_path) != export_sha:
            raise ValueError("frontend generator export manifest SHA mismatch")
        try:
            export_obj = json.loads(export_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid bundled generator export manifest: {exc}") from exc
        if (export_obj.get("schema"), export_obj.get("status"), export_obj.get("target"), export_obj.get("full_model_sha256"), export_obj.get("output_sha256")) != ("kokoro-v10-generator-export-1", "PASS", "rk3588", FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256):
            raise ValueError("frontend bundled generator export identity mismatch")
        model_rows = (("probe_model", "duration_probe.onnx"), ("target_model", "target_frontend.onnx")) if schema == "kokoro-v10-long32-frontend-2" else (("stem_model", "frontend_stem.onnx"), ("duration_model", "duration_head.onnx"), ("target_model", "target_tail.onnx"))
        for key, expected_name in model_rows:
            if manifest.get(key) != expected_name:
                raise ValueError(f"frontend manifest {key} mismatch")
            path = self.model_dir / expected_name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or symlinked frontend model: {expected_name}")
            expected_sha = manifest.get({"probe_model": "probe_sha256", "stem_model": "stem_sha256", "duration_model": "duration_sha256", "target_model": "target_sha256"}[key])
            if not isinstance(expected_sha, str) or self._sha256(path) != expected_sha:
                raise ValueError(f"frontend model SHA mismatch: {expected_name}")
        assets = manifest.get("assets")
        if not isinstance(assets, dict):
            raise ValueError("frontend manifest assets missing")
        for name in assets:
            candidate = Path(str(name))
            if candidate.name != name or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"frontend asset path traversal: {name}")
        voice_names = set(VOICES.values())
        required = {"config.json", "tokens.json", "vocab.json"} & set(assets)
        if not required:
            raise ValueError("frontend manifest must include config/tokens/vocab asset")
        required |= {name + ".bin" for name in voice_names}
        required.add(str(export_meta.get("path", "")))
        if not required.issubset(assets):
            raise ValueError("frontend manifest missing voice or vocabulary asset")
        for name in required:
            # Manifest names are bundle-local basenames, never paths.
            if Path(name).name != name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"frontend asset path traversal: {name}")
            path = self.model_dir / name
            row = assets[name]
            if path.is_symlink() or not path.is_file() or not isinstance(row, dict):
                raise ValueError(f"missing or symlinked frontend asset: {name}")
            if row.get("bytes") != path.stat().st_size or row.get("sha256") != self._sha256(path):
                raise ValueError(f"frontend asset hash/size mismatch: {name}")
        self._manifest_schema = schema

    def _load_vocab(self) -> dict[str, int]:
        for name in ("config.json", "tokens.json", "vocab.json"):
            p = self.model_dir / name
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if name == "config.json":
                    data = data.get("vocab", data.get("token_to_id"))
                else:
                    data = data.get("vocab", data.get("token_to_id", data))
                if isinstance(data, dict) and data:
                    try:
                        result = {str(k): int(v) for k, v in data.items()}
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"invalid Kokoro vocabulary in {p.name}") from exc
                    if result:
                        return result
        raise FileNotFoundError(f"Kokoro vocabulary not found in {self.model_dir}")

    @staticmethod
    def _official_g2p(route: str) -> Callable[[str], str]:
        try:
            es = importlib.import_module("misaki.espeak").EspeakFallback
            if route in {"a", "b"}:
                en = importlib.import_module("misaki.en"); british = route == "b"
                obj = en.G2P(trf=False, british=british, fallback=es(british=british), unk="")
            elif route == "j": obj = importlib.import_module("misaki.ja").JAG2P()
            elif route == "z":
                en = importlib.import_module("misaki.en"); zh = importlib.import_module("misaki.zh")
                enobj = en.G2P(trf=False, british=False, fallback=es(british=False), unk="")
                obj = zh.ZHG2P(version=None, unk="", en_callable=lambda text: enobj(text)[0])
            else:
                cls = importlib.import_module("misaki.espeak").EspeakG2P
                lang = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}[route]
                obj = cls(language=lang)
        except Exception as exc:
            raise RuntimeError(f"G2P unavailable for route {route}: {exc}") from exc
        def call(text: str) -> str:
            out = obj(text); out = out[0] if isinstance(out, tuple) else out
            if not isinstance(out, str): raise TypeError("G2P did not return phoneme string")
            return out
        return call

    def _route(self, language: str) -> str:
        key = str(language).strip().lower()
        if key not in LANG_ALIASES: raise ValueError(f"unknown Kokoro language/route: {language}")
        return LANG_ALIASES[key]

    def _voice(self, route: str, n: int) -> np.ndarray:
        with self._init_lock:
            cached = self._voice_cache.get(route)
            if cached is None:
                p = self.model_dir / (VOICES[route] + ".bin")
                if not p.is_file() or p.stat().st_size != int(np.prod(VOICE_SHAPE)) * 4: raise ValueError(f"invalid/missing voice blob: {p}")
                voice = np.fromfile(p, dtype="<f4").reshape(VOICE_SHAPE)
                if not np.isfinite(voice).all(): raise ValueError("voice ABI/finite gate failed")
                voice = np.array(voice, dtype=np.float32, copy=True); voice.setflags(write=False)
                self._voice_cache[route] = voice
                cached = voice
        if not 0 <= n < 510: raise ValueError("voice ABI/finite gate failed")
        return cached[n]

    def _g2p(self, route: str) -> tuple[Callable[[str], str], threading.Lock]:
        with self._g2p_lock:
            if self.g2p_factory is not self._g2p_factory_seen:
                self._g2p_cache.clear(); self._g2p_call_locks.clear(); self._g2p_factory_seen = self.g2p_factory
            fn = self._g2p_cache.get(route)
            if fn is None:
                fn = self.g2p_factory(route)
                if not callable(fn): raise TypeError("G2P factory must return callable")
                self._g2p_cache[route] = fn
                self._g2p_call_locks[route] = threading.Lock()
            return fn, self._g2p_call_locks[route]

    def preload_sessions(self) -> Mapping[str, float]:
        """Atomically build v2's two sessions or v3's three sessions."""
        if self._legacy_sessions or (self._manifest_schema == "kokoro-v10-long32-frontend-2" and self.duration_head is not None and self.target_tail is not None) or (self._manifest_schema == "kokoro-v10-long32-frontend-3" and self._split_ready): return {}
        started = time.perf_counter()
        with self._init_lock:
            if self._legacy_sessions or (self._manifest_schema == "kokoro-v10-long32-frontend-2" and self.duration_head is not None and self.target_tail is not None) or (self._manifest_schema == "kokoro-v10-long32-frontend-3" and self._split_ready): return {}
            import onnxruntime as ort
            options = self._session_options
            kwargs = {"providers": ["CPUExecutionProvider"]}
            if options is not None: kwargs["sess_options"] = options
            stem, duration, target = self.stem, self.duration_head, self.target_tail
            if self._manifest_schema == "kokoro-v10-long32-frontend-2":
                if duration is None: duration = ort.InferenceSession(str(self.model_dir / "duration_probe.onnx"), **kwargs)
                if target is None: target = ort.InferenceSession(str(self.model_dir / "target_frontend.onnx"), **kwargs)
            else:
                if stem is None: stem = ort.InferenceSession(str(self.model_dir / "frontend_stem.onnx"), **kwargs)
                if duration is None: duration = ort.InferenceSession(str(self.model_dir / "duration_head.onnx"), **kwargs)
                if target is None: target = ort.InferenceSession(str(self.model_dir / "target_tail.onnx"), **kwargs)
            if self._manifest_schema == "kokoro-v10-long32-frontend-3":
                names = self._resolve_split_input_names(stem, duration, target)
            else:
                names = None
            if names is not None:
                self._publish_split_bundle(stem, duration, target, names)
            else:
                self.stem, self.duration_head, self.target_tail = stem, duration, target
                self.probe, self.target = duration, target
        return {"session_init": (time.perf_counter() - started) * 1000.0}

    @staticmethod
    def _session_inputs(session: Any, label: str) -> list[Any]:
        getter = getattr(session, "get_inputs", None)
        if not callable(getter):
            # Lightweight injected test doubles do not expose ORT metadata.
            # Real sessions are always ORT sessions and are checked below.
            return []
        inputs = list(getter())
        if not inputs:
            raise RuntimeError(f"{label} session has no inputs")
        return inputs

    def _publish_split_bundle(self, stem: Any, duration: Any, target: Any, names: Mapping[str, str]) -> None:
        self.stem, self.duration_head, self.target_tail = stem, duration, target
        self.probe, self.target = duration, target
        self._duration_text_name = names["duration_text"]
        self._duration_style_name = names["duration_style"]
        self._target_text_name = names["target_text"]
        self._target_stem_style_name = names["target_stem_style"]
        self._split_ready = True

    def _resolve_split_input_names(self, stem: Any, duration: Any, target: Any) -> Mapping[str, str]:
        """Resolve exported split-graph names from the deployed ORT ABI.

        onnx extraction can retain internal tensor names instead of the names
        requested by the exporter. Match by the strict shape/type contract so
        either form is accepted, while an ambiguous or incompatible graph
        fails closed before the service can accept requests.
        """
        sin = self._session_inputs(stem, "frontend_stem")
        din = self._session_inputs(duration, "duration_head")
        tin = self._session_inputs(target, "target_tail")
        if not sin and not din and not tin:
            return {"duration_text": STEM_TEXT, "duration_style": STEM_STYLE,
                    "target_text": STEM_TEXT, "target_stem_style": STEM_STYLE}

        def typ(x: Any) -> str:
            return str(getattr(x, "type", ""))

        def shape(x: Any) -> list[Any]:
            return list(getattr(x, "shape", []))

        def is_1n(value: list[Any]) -> bool:
            return (len(value) == 2 and isinstance(value[0], int) and not isinstance(value[0], bool)
                    and value[0] == 1 and (isinstance(value[1], str) or value[1] is None))

        def unique(inputs: list[Any], label: str) -> None:
            names = [getattr(item, "name", None) for item in inputs]
            if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
                raise RuntimeError(f"{label} duplicate/invalid NodeArg names")

        unique(sin, "frontend_stem")
        unique(din, "duration_head")
        unique(tin, "target_tail")

        if len(sin) != 2:
            raise RuntimeError(f"frontend_stem input count mismatch: {len(sin)}")
        sin_named = {x.name: x for x in sin}
        if set(sin_named) != {"input_ids", "style"}:
            raise RuntimeError("frontend_stem input names mismatch")
        if typ(sin_named["input_ids"]) != "tensor(int64)" or not is_1n(shape(sin_named["input_ids"])):
            raise RuntimeError("frontend_stem input_ids ABI mismatch")
        if str(getattr(sin_named["style"], "type", "")) != "tensor(float)" or list(getattr(sin_named["style"], "shape", [])) != [1, 256]:
            raise RuntimeError("frontend_stem style ABI mismatch")
        if len(din) != 3 or len(tin) != 5:
            raise RuntimeError(f"split frontend input count mismatch: duration={len(din)} target={len(tin)}")

        speed = [x for x in din if x.name == "speed" and typ(x) == "tensor(float)" and shape(x) == [1]]
        if len(speed) != 1:
            raise RuntimeError("duration_head speed input ABI mismatch")
        dfloat = [x for x in din if x not in speed and typ(x) == "tensor(float)"]
        if len(dfloat) != 2:
            raise RuntimeError("duration_head stem input ABI mismatch")
        dtext = [x for x in dfloat if len(shape(x)) == 3 and shape(x)[-1] == 640]
        dstyle = [x for x in dfloat if shape(x) == [1, 128]]
        if len(dtext) != 1 or len(dstyle) != 1:
            raise RuntimeError("duration_head stem shape ABI mismatch")

        required = {"style", "target_durations", "input_ids"}
        named = {x.name: x for x in tin}
        if not required.issubset(named):
            raise RuntimeError("target_tail control input ABI mismatch")
        if typ(named["style"]) != "tensor(float)" or shape(named["style"]) != [1, 256]:
            raise RuntimeError("target_tail style ABI mismatch")
        for control in ("target_durations", "input_ids"):
            if typ(named[control]) != "tensor(int64)" or not is_1n(shape(named[control])):
                raise RuntimeError(f"target_tail {control} ABI mismatch")
        control_dims = [shape(named[name])[1] for name in ("target_durations", "input_ids")]
        if control_dims[0] is not None and control_dims[1] is not None and control_dims[0] != control_dims[1]:
            raise RuntimeError("target_tail dynamic sequence dimension mismatch")
        if typ(named["target_durations"]) != "tensor(int64)" or typ(named["input_ids"]) != "tensor(int64)":
            raise RuntimeError("target_tail integer input ABI mismatch")
        tfloat = [x for x in tin if x.name not in required and typ(x) == "tensor(float)"]
        if len(tfloat) != 2:
            raise RuntimeError("target_tail stem input ABI mismatch")
        ttext = [x for x in tfloat if len(shape(x)) == 3 and shape(x)[-1] == 640]
        tstyle = [x for x in tfloat if shape(x) == [1, 128]]
        if len(ttext) != 1 or len(tstyle) != 1:
            raise RuntimeError("target_tail stem shape ABI mismatch")
        return {"duration_text": dtext[0].name, "duration_style": dstyle[0].name,
                "target_text": ttext[0].name, "target_stem_style": tstyle[0].name}

    def _ids(self, phonemes: str) -> list[int]:
        ids = []; unknown = []
        for ch in phonemes:
            key = " " if ch.isspace() else ch
            if key not in self.vocab: unknown.append(ch)
            else: ids.append(self.vocab[key])
        if unknown: raise ValueError(f"unknown/dropped phoneme(s): {unknown!r}")
        if not ids: raise ValueError("phoneme length outside 1..200: 0")
        if len(ids) > 200: raise PhonemeLengthError(f"phoneme length outside 1..200: {len(ids)}")
        return ids

    def prepare(self, text: str, language: str, speed: float = 1.0, *, cancel_check=None) -> Prepared:
        check = cancel_check if cancel_check is not None else lambda: None
        check()
        route = self._route(language)
        if not isinstance(text, str) or not text.strip(): raise ValueError("text must be non-empty")
        speed_a = np.array([speed], dtype=np.float32, copy=True); speed_a.setflags(write=False)
        if not np.isfinite(speed_a).all() or speed <= 0: raise ValueError("speed must be finite and > 0")
        detail: dict[str, float] = {}
        t0 = time.perf_counter(); fn, call_lock = self._g2p(route); detail["g2p_factory"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        with call_lock: ph = fn(text)
        check()
        detail["g2p"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); ids = self._ids(ph); detail["ids"] = (time.perf_counter() - t0) * 1000
        tok = np.array([[0, *ids, 0]], dtype=np.int64, copy=True); tok.setflags(write=False)
        t0 = time.perf_counter(); style = np.array(self._voice(route, len(ids)), dtype=np.float32, copy=True); style.setflags(write=False); detail["voice"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); init = self.preload_sessions(); detail.update(init)
        check()
        if self._manifest_schema == "kokoro-v10-long32-frontend-3" and not self._split_ready:
            raise RuntimeError("schema-3 frontend sessions are not ready")
        detail.setdefault("session_init", 0.0)
        t0 = time.perf_counter()
        if self._legacy_sessions or self._manifest_schema == "kokoro-v10-long32-frontend-2":
            t0 = time.perf_counter()
            d = np.asarray(self.duration_head.run(["durations"], {"input_ids": tok, "style": style, "speed": speed_a})[0])
            detail["probe"] = (time.perf_counter() - t0) * 1000
            stem_text = stem_style = None
        else:
            t0 = time.perf_counter()
            stem_out = self.stem.run(["stem_text", "stem_style"], {"input_ids": tok, "style": style})
            check()
            stem_text, stem_style = map(np.asarray, stem_out)
            stem_text = np.array(stem_text, dtype=np.float32, copy=True); stem_text.setflags(write=False)
            stem_style = np.array(stem_style, dtype=np.float32, copy=True); stem_style.setflags(write=False)
            detail["stem"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            d = np.asarray(self.duration_head.run(["durations"], {self._duration_text_name: stem_text, self._duration_style_name: stem_style, "speed": speed_a})[0])
            detail["duration_head"] = (time.perf_counter() - t0) * 1000
        check()
        if d.dtype != np.int64 or d.shape != (1, len(ids) + 2) or np.any(d < 1): raise ValueError(f"natural duration ABI failed: {d.dtype}/{d.shape}")
        d = np.array(d.reshape(-1), dtype=np.int64, copy=True); d.setflags(write=False)
        return Prepared(route, ROUTES[route], text, tok, style, speed_a, d, int(d.sum()), stem_text, stem_style, detail)

    def render(self, prepared: Prepared, target_D: int) -> dict[str, Any]:
        if not isinstance(prepared, Prepared): raise TypeError("render expects Prepared")
        t0 = time.perf_counter(); durations = snap_durations(prepared.natural_durations, int(target_D)); snap_ms = (time.perf_counter() - t0) * 1000
        if int(durations.sum()) != int(target_D): raise RuntimeError("target duration sum mismatch")
        init = self.preload_sessions()
        if self._manifest_schema == "kokoro-v10-long32-frontend-3" and not self._split_ready:
            raise RuntimeError("schema-3 frontend sessions are not ready")
        t0 = time.perf_counter()
        if self._legacy_sessions or self._manifest_schema == "kokoro-v10-long32-frontend-2":
            outputs = self.target_tail.run(["x", "s", "har"], {"input_ids": prepared.input_ids, "style": prepared.style, "speed": prepared.speed, "target_durations": durations[None, :]})
        else:
            outputs = self.target_tail.run(["x", "s", "har"], {self._target_text_name: prepared.stem_text, self._target_stem_style_name: prepared.stem_style, "style": prepared.style, "target_durations": durations[None, :], "input_ids": prepared.input_ids})
        target_ms = (time.perf_counter() - t0) * 1000
        x, s, har = map(np.asarray, outputs); t = 2 * int(target_D); expected = ((1, 512, t), (1, 128), (1, 22, 60 * t + 1))
        if tuple(x.shape) != expected[0] or tuple(s.shape) != expected[1] or tuple(har.shape) != expected[2] or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (x, s, har)): raise ValueError("target frontend shape/finite gate failed")
        detail = dict(prepared.timings or {}); detail.update(init); detail["target_tail"] = target_ms; detail["snap"] = snap_ms
        return {"x": x, "s": s, "har": har, "durations": durations, "T": t, "F": 60 * t + 1, "route": prepared.route, "language": prepared.language, "frontend_detail": detail}
