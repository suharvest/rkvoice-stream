"""Production runtime for the Kokoro v1.0 long32 RKNN bundle.

The frontend is deliberately injected.  This module only owns the validated
``x/s/har`` ABI, the 32-context generator and the CPU post-iSTFT tail.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import logging
import operator
import os
import signal
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

logger = logging.getLogger(__name__)
PLATFORM_TS = {
    "rk3588": (320, 400, 480, 560, 640),
    # RKNN Toolkit 2.3.2 SIGILLs while compiling the rb345 F-shaped convs at
    # T400/T560 on RK3576.  These six device-proven compiler shapes retain
    # 4--8 second coverage with <10% nearest-profile duration error.
    "rk3576": (320, 368, 416, 480, 576, 640),
}
TS = PLATFORM_TS["rk3588"]
RB = (3, 4, 5)
STAGES = ("main0", "noise0", "rb0", "rb1", "rb2", "main1", "post")
FULL_MODEL_SHA256 = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
GENERATOR_SOURCE_SHA256 = "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c"
SOURCE_SHA256 = GENERATOR_SOURCE_SHA256
CONTEXT_STAGES = STAGES + tuple(
    f"rb{block}-convs{half}_{index}"
    for block in RB for half in (1, 2) for index in range(3)
) + ("noise1-preconv",) + tuple(
    f"noise1-convs{half}_{index}" for half in (1, 2) for index in range(3)
)
DEFAULT_MAX_SNAP_RATIO = 0.10
MAX_SNAP_RATIO = 1.0

class Long32Fallback(RuntimeError):
    """Sentence must be handled by the legacy hybrid backend."""


class Long32Cancelled(RuntimeError):
    """Internal control flow for a cancelled long32 request."""


def _cancelled(cancel_event: object | None) -> bool:
    return bool(cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _validated_profiles(profiles: Any, platform: str) -> tuple[int, ...]:
    """Validate an operator route without weakening the platform manifest gate."""
    try:
        allowed = tuple(PLATFORM_TS[platform])
    except KeyError as exc:
        raise ValueError(f"unsupported long32 platform: {platform}") from exc
    try:
        values = tuple(operator.index(value) for value in profiles)
    except (TypeError, ValueError) as exc:
        raise ValueError("route profiles must be integer T values") from exc
    if not values:
        raise ValueError("route profiles must be non-empty")
    if any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError("route profiles must be strictly ascending and unique")
    if not set(values).issubset(allowed):
        raise ValueError(f"route profiles must be a subset of {platform} PLATFORM_TS")
    return values


def parse_route_ts(value: str | None, *, platform: str = "rk3588") -> tuple[int, ...]:
    """Parse ``KOKORO_LONG32_ROUTE_TS`` with a fail-closed grammar.

    An unset or genuinely empty variable means all platform profiles for
    backward compatibility.  Any non-empty value must be ``T,T,...`` with no
    duplicate or out-of-order values; whitespace and empty list members are
    rejected rather than silently normalized.
    """
    allowed = tuple(PLATFORM_TS.get(platform, ()))
    if not allowed:
        raise ValueError(f"unsupported long32 platform: {platform}")
    if value is None or value == "":
        return allowed
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError("KOKORO_LONG32_ROUTE_TS must be comma-separated integers")
    return _validated_profiles(tuple(int(item) for item in value.split(",")), platform)


def parse_max_snap_ratio(value: str | float | int | None = None) -> float:
    """Parse the duration snap guard without accepting non-finite values."""
    if value is None:
        return DEFAULT_MAX_SNAP_RATIO
    if isinstance(value, bool) or (isinstance(value, str) and value == ""):
        raise ValueError("KOKORO_LONG32_MAX_SNAP_RATIO must be finite and in [0, 1]")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("KOKORO_LONG32_MAX_SNAP_RATIO must be finite and in [0, 1]") from exc
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > MAX_SNAP_RATIO:
        raise ValueError("KOKORO_LONG32_MAX_SNAP_RATIO must be finite and in [0, 1]")
    return ratio


def select_profile(natural_d: int, *, max_ratio: float = 0.10, platform: str = "rk3588", profiles: Any | None = None) -> dict[str, Any]:
    """Choose the nearest 4--8 second profile and report duration snapping."""
    max_ratio = parse_max_snap_ratio(max_ratio)
    d = int(natural_d)
    if d <= 0:
        raise ValueError("natural_d must be positive")
    route_profiles = _validated_profiles(PLATFORM_TS.get(platform, ()) if profiles is None else profiles, platform)
    target = min(route_profiles, key=lambda t: (abs(t // 2 - d), t))
    target_d = target // 2
    ratio = target_d / d
    result = {"natural_D": d, "target_D": target_d, "T": target, "F": 60 * target + 1,
              "ratio": ratio, "snap_ratio": abs(ratio - 1.0),
              "route_ts": list(route_profiles), "fallback": abs(ratio - 1.0) > max_ratio}
    if result["fallback"]:
        result["reason"] = "duration ratio exceeds configured maximum"
    return result


def snap_durations(durations: Any, target_d: int) -> np.ndarray:
    """Snap positive frontend durations while preserving integer total D."""
    d = np.asarray(durations, dtype=np.int64).reshape(-1)
    if d.size == 0 or np.any(d < 1) or int(target_d) < d.size:
        raise ValueError("durations must be non-empty positive values and target_D >= N")
    target_d = int(target_d)
    if int(d.sum()) == target_d:
        return d.copy()
    excess = d - 1
    total = int(excess.sum())
    wanted = target_d - d.size
    if total == 0:
        out = np.ones_like(d); out[0] += wanted
        return out
    raw = excess.astype(np.float64) * wanted / total
    out = np.ones_like(d) + np.floor(raw).astype(np.int64)
    remain = wanted - int((out - 1).sum())
    order = np.argsort(-(raw - np.floor(raw)), kind="stable")
    for i in order[:remain]: out[i] += 1
    return out


def _finite(a: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype != np.float32 or len(x.shape) != len(shape) or any(e is not None and e != g for e, g in zip(shape, x.shape)) or not np.isfinite(x).all():
        raise ValueError(f"{name} ABI mismatch: expected float32 {shape}, got {x.dtype} {x.shape}")
    return np.ascontiguousarray(x)


class _Context:
    def __init__(self, lite: Any, path: Path, ins: list[tuple[int | None, ...]], out: tuple[int | None, ...], name: str, core_mask: int = 0, *, explicit_mask: bool | None = None):
        self.obj, self.path, self.ins, self.out, self.name = lite, path, ins, out, name
        self.released = False
        if isinstance(core_mask, bool) or not isinstance(core_mask, int) or core_mask not in (0, 1, 2, 4):
            raise ValueError(f"{name}: invalid core mask")
        self.core_mask = core_mask
        self.explicit_mask = bool(core_mask) if explicit_mask is None else explicit_mask
        if self.explicit_mask != bool(core_mask):
            raise ValueError(f"{name}: explicit core-mask contradicts mask")
        try:
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            if lite.load_rknn(str(path)) != 0:
                raise RuntimeError(f"{name}: RKNN load failed")
            if self.explicit_mask:
                ret = lite.init_runtime(core_mask=core_mask)
            else:
                try:
                    ret = lite.init_runtime(core_mask=getattr(lite, "NPU_CORE_AUTO", 0))
                except TypeError:
                    ret = lite.init_runtime()
            if ret != 0:
                raise RuntimeError(f"{name}: RKNN load/init failed")
        except Exception as load_error:
            try:
                self.release()
            except Exception as cleanup_error:
                combined = RuntimeError(
                    f"{name}: load/init failed: {type(load_error).__name__}: {load_error}; "
                    f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
                combined.unreleased_context = self
                raise combined from load_error
            raise
    def run(self, values: list[np.ndarray]) -> np.ndarray:
        if len(values) != len(self.ins): raise ValueError(f"{self.name}: input count mismatch")
        for x, shape in zip(values, self.ins): _finite(x, shape, self.name + " input")
        out = self.obj.inference(inputs=[np.ascontiguousarray(x) for x in values])
        if not isinstance(out, (list, tuple)) or len(out) != 1: raise RuntimeError(f"{self.name}: output count")
        return _finite(out[0], self.out, self.name + " output")
    def release(self) -> None:
        if not self.released:
            self.obj.release()
            self.released = True


class _Branch:
    def __init__(self, lite_cls: Any, root: Path, block: int, core_masks: Mapping[str, int] | None = None, *, load_contexts: bool = True):
        if type(load_contexts) is not bool:
            raise TypeError("load_contexts must be a bool")
        models, params = root / "models", root / "params" / "rb345"
        self.ctx: list[_Context] = []
        self.params = []
        try:
            for half in (1, 2):
                for stage in range(3):
                    n = f"rb{block}-convs{half}_{stage}.fp16.rknn"
                    if load_contexts:
                        self.ctx.append(_Context(lite_cls(), models / n, [(1, 128, None)], (1, 128, None), n, (core_masks or {}).get(n.removesuffix('.fp16.rknn'), 0)))
                    stem = params / f"rb{block}-adain{half}_{stage}"
                    self.params.append((np.fromfile(str(stem)+"-fc-weight.bin", np.float32).reshape(256,128),
                        np.fromfile(str(stem)+"-fc-bias.bin", np.float32).reshape(256),
                        np.fromfile(str(stem)+"-alpha.bin", np.float32).reshape(1,128,1)))
        except Exception as load_error:
            partial = getattr(load_error, "unreleased_context", None)
            if partial is not None and partial not in self.ctx:
                self.ctx.append(partial)
            try:
                self.release()
            except Exception as cleanup_error:
                combined = RuntimeError(
                    f"rb{block} load failed: {type(load_error).__name__}: {load_error}; "
                    f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
                combined.unreleased_component = self
                raise combined from load_error
            raise
    def run(self, x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, dict[str,float]]:
        if not self.ctx:
            raise RuntimeError("params-only branch cannot run")
        z = x.copy(); total = 0.0
        for stage in range(3):
            residual = z.copy()
            for half in (1, 2):
                w,b,a = self.params[(half-1)*3+stage]
                m = z.mean(2, keepdims=True); c = z-m; z = c/np.sqrt((c*c).mean(2,keepdims=True)+1e-5)
                style = s @ w.T + b
                z = z*(style[:,:128,None]+1)+style[:,128:,None]
                z = z+(np.sin(a*z)**2)/a
                started = time.perf_counter(); z = self.ctx[(half-1)*3+stage].run([z]); total += (time.perf_counter()-started)*1000
            z = np.ascontiguousarray(residual+z, np.float32)
        return z, {"conv_ms": total}
    def release(self):
        errors=[]
        retained=[]
        for c in reversed(self.ctx):
            try: c.release()
            except Exception as e:
                errors.append(str(e)); retained.append(c)
        self.ctx = list(reversed(retained))
        if errors: raise RuntimeError("branch release: "+";".join(errors))


class _Noise1:
    def __init__(self, lite_cls: Any, root: Path, core_masks: Mapping[str, int] | None = None):
        self.ctx=[]; self.params=[]; p=root/"params"/"noise1"; m=root/"models"
        try:
            names=["noise1-preconv.fp16.rknn"]+[f"noise1-convs{h}_{i}.fp16.rknn" for h in (1,2) for i in range(3)]
            for i,n in enumerate(names): self.ctx.append(_Context(lite_cls(),m/n,[(1,22,None) if i==0 else (1,128,None)],(1,128,None),n, (core_masks or {}).get(n.removesuffix('.fp16.rknn'), 0)))
            for h in (1,2):
                for i in range(3):
                    stem=p/f"noise1-adain{h}_{i}"; self.params.append((np.fromfile(str(stem)+"-fc-weight.bin",np.float32).reshape(256,128),np.fromfile(str(stem)+"-fc-bias.bin",np.float32).reshape(256),np.fromfile(str(stem)+"-alpha.bin",np.float32).reshape(1,128,1)))
        except Exception as load_error:
            partial = getattr(load_error, "unreleased_context", None)
            if partial is not None and partial not in self.ctx:
                self.ctx.append(partial)
            try:
                self.release()
            except Exception as cleanup_error:
                combined = RuntimeError(
                    f"noise1 load failed: {type(load_error).__name__}: {load_error}; "
                    f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
                combined.unreleased_component = self
                raise combined from load_error
            raise
    def run(self, har, s):
        z=self.ctx[0].run([har]);
        for stage in range(3):
            residual=z.copy()
            for half in (1,2):
                w,b,a=self.params[(half-1)*3+stage]; m=z.mean(2,keepdims=True); c=z-m; z=c/np.sqrt((c*c).mean(2,keepdims=True)+1e-5); style=s@w.T+b; z=z*(style[:,:128,None]+1)+style[:,128:,None]; z=z+(np.sin(a*z)**2)/a; z=self.ctx[1+(half-1)*3+stage].run([z])
            z=np.ascontiguousarray(residual+z,np.float32)
        return z
    def release(self):
        errors=[]
        retained=[]
        for c in reversed(self.ctx):
            try:
                c.release()
            except Exception as exc:
                errors.append(str(exc)); retained.append(c)
        self.ctx = list(reversed(retained))
        if errors:
            raise RuntimeError("noise1 release: " + "; ".join(errors))


def _istft(conv: np.ndarray) -> np.ndarray:
    from .kokoro_istft import conv_post_to_waveform
    return conv_post_to_waveform(conv)


class KokoroLong32Runtime:
    def __init__(self, model_dir: str | Path, *, frontend: Callable[..., Mapping[str, Any]] | None = None, lite_cls: Any | None = None, platform: str | None = None, context_core_masks: Mapping[str, int] | None = None, route_ts: Any | None = None, prefix_only: bool = False):
        if type(prefix_only) is not bool:
            raise TypeError("prefix_only must be a bool")
        if context_core_masks is not None and not isinstance(context_core_masks, Mapping):
            raise TypeError("context_core_masks must be a Mapping")
        masks = dict(context_core_masks or {})
        unknown = set(masks) - set(CONTEXT_STAGES)
        if unknown: raise ValueError(f"unknown long32 context core-mask keys: {sorted(unknown)}")
        for key, value in masks.items():
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2, 4):
                raise ValueError(f"invalid core mask for {key}: expected 0, 1, 2, or 4")
        self.root=Path(model_dir); self.frontend=frontend; self.lite_cls=lite_cls; self.platform=platform or os.environ.get("KOKORO_RKNN_PLATFORM","rk3588"); self.prefix_only=prefix_only; self.route_ts=parse_route_ts(os.environ.get("KOKORO_LONG32_ROUTE_TS"), platform=self.platform) if route_ts is None else _validated_profiles(route_ts, self.platform); self.context_core_masks={key:masks.get(key,0) for key in CONTEXT_STAGES}; self.contexts={}; self.branches={}; self.noise=None; self.noise0_profiles={}; self._noise0_lock=threading.RLock(); self._lifecycle_lock=threading.RLock(); self._noise0_specs={}; self.pool=None; self.ready=False; self._cleanup_registered=False
    def _manifest(self):
        path=self.root/"build-manifest.json"
        if not path.is_file() or path.is_symlink(): raise ValueError("long32 build-manifest.json missing")
        obj=json.loads(path.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if obj.get("schema") != f"kokoro-v10-{self.platform}-long-hybrid-full-v7" or obj.get("status")!="PASS" or obj.get("contexts")!=32 or obj.get("target")!=self.platform: raise ValueError("long32 manifest schema/status/context/platform mismatch")
        if (obj.get("full_source_sha256"), obj.get("generator_source_sha256")) != (FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256):
            raise ValueError("long32 dual source SHA mismatch")
        if (obj.get("source_sha256"), obj.get("source_sha256_deprecated"), obj.get("source_sha256_kind")) != (GENERATOR_SOURCE_SHA256, True, "generator_model"):
            raise ValueError("long32 deprecated source SHA mismatch")
        manifest_sha = obj.get("generator_export_manifest_sha256")
        if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
            raise ValueError("long32 generator export manifest SHA missing")
        preflight = obj.get("preflight_evidence")
        if not isinstance(preflight, dict) or preflight.get("generator_export_manifest_sha256") != manifest_sha:
            raise ValueError("long32 preflight generator export chain mismatch")
        export_path = self.root / "generator-export.manifest.json"
        if export_path.is_symlink() or not export_path.is_file() or _sha(export_path) != manifest_sha:
            raise ValueError("long32 generator export manifest SHA mismatch")
        export_obj = json.loads(export_path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if (export_obj.get("schema"), export_obj.get("status"), export_obj.get("target"), export_obj.get("full_model_sha256"), export_obj.get("output_sha256")) != ("kokoro-v10-generator-export-1", "PASS", "rk3588", FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256):
            raise ValueError("long32 generator export manifest identity mismatch")
        if obj.get("precision")!="FP16" or obj.get("toolkit")!="2.3.2": raise ValueError("long32 manifest precision/toolkit mismatch")
        rows=obj.get("rows"); profiles=obj.get("profiles")
        if not isinstance(rows,list) or len(rows)!=32 or not isinstance(profiles,dict): raise ValueError("long32 manifest rows/profiles mismatch")
        names = []
        stages = []
        expected_profiles = PLATFORM_TS.get(self.platform)
        if expected_profiles is None or set(profiles) != {f"T{t}" for t in expected_profiles}:
            raise ValueError("long32 manifest platform profile set mismatch")
        for expected_t in expected_profiles:
            key = f"T{expected_t}"
            p = profiles.get(key)
            if not isinstance(p, dict) or (p.get("T"),p.get("L"),p.get("F")) != (expected_t,10*expected_t,60*expected_t+1): raise ValueError("long32 profile ABI mismatch")
        for row in rows:
            if not isinstance(row,dict) or row.get("status")!="PASS": raise ValueError("long32 manifest row status mismatch")
            # RK3576 noise0 is compiled as one static graph per profile.  The
            # profile map is mandatory and is deliberately not allowed to
            # fall back to a neighbouring T shape.
            if self.platform == "rk3576" and row.get("stage") == "noise0":
                static = row.get("static_profiles")
                expected_ts = tuple(expected_profiles)
                if not isinstance(static, dict) or set(static) != {f"T{t}" for t in expected_ts}:
                    raise ValueError("RK3576 noise0 static profile set mismatch")
                for t in expected_ts:
                    spec = static[f"T{t}"]
                    if not isinstance(spec, dict) or (spec.get("T"), spec.get("F"), spec.get("L")) != (t, 60*t+1, 10*t):
                        raise ValueError("RK3576 noise0 static profile ABI mismatch")
                    if (spec.get("inputs"), spec.get("outputs"), spec.get("input_count"), spec.get("output_count")) != ([[1,128],[1,22,60*t+1]], [[1,256,10*t]], 2, 1):
                        raise ValueError("RK3576 noise0 static profile tensor ABI mismatch")
                    raw_text = spec.get("rknn", "")
                    if raw_text != f"noise0-T{t}.fp16.rknn":
                        raise ValueError(f"RK3576 noise0 static basename mismatch: T{t}")
                    raw = Path(raw_text).name
                    expected = spec.get("rknn_sha256")
                    model = self.root / "models" / raw
                    if not raw or not expected or not model.is_file() or model.is_symlink() or _sha(model) != expected:
                        raise ValueError(f"long32 exact static noise0 model/SHA mismatch: T{t}")
                    self._noise0_specs[t] = (model, expected)
                names.append("noise0.static_profiles")
                stages.append(row.get("stage"))
                continue
            raw_name = row.get("rknn", row.get("model"))
            if not raw_name and row.get("stage"):
                raw_name = f"{row['stage']}.fp16.rknn"
            name=Path(str(raw_name or "")).name; p=self.root/"models"/name
            names.append(name)
            stages.append(row.get("stage"))
            expected=row.get("rknn_sha256",row.get("sha256"))
            if not name or not p.is_file() or not expected or _sha(p)!=expected: raise ValueError(f"long32 exact model/SHA mismatch: {name}")
        if len(set(names)) != 32: raise ValueError("long32 manifest must contain 32 unique contexts")
        if set(stages) != set(CONTEXT_STAGES) or len(set(stages)) != len(CONTEXT_STAGES):
            raise ValueError("long32 manifest stage set mismatch")
        if self.platform == "rk3576":
            self._device_smoke(path, obj)
        self.manifest=obj; return obj

    def _device_smoke(self, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        path = self.root / "device-smoke-report.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("RK3576 device smoke report missing")
        report = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        profiles = list(PLATFORM_TS["rk3576"])
        required = 32 * len(profiles)
        if (report.get("schema"), report.get("status"), report.get("target")) != ("kokoro-v10-isolated-smoke-2", "PASS", "rk3576"):
            raise ValueError("RK3576 device smoke identity/status mismatch")
        if (report.get("full_source_sha256"), report.get("generator_source_sha256"), report.get("generator_export_manifest_sha256")) != (FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256, manifest.get("generator_export_manifest_sha256")):
            raise ValueError("RK3576 device smoke dual provenance mismatch")
        if report.get("build_manifest_sha256") != _sha(manifest_path):
            raise ValueError("RK3576 device smoke build-manifest SHA mismatch")
        if report.get("required_runs") != required or report.get("passed_runs") != required or report.get("profiles") != profiles:
            raise ValueError("RK3576 device smoke coverage mismatch")
        expected = {row["stage"]: row for row in manifest["rows"]}
        models = report.get("models")
        if not isinstance(models, list) or len(models) != 32:
            raise ValueError("RK3576 device smoke model count mismatch")
        seen: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("RK3576 device smoke model entry mismatch")
            stage = model.get("stage")
            if stage in seen or stage not in expected:
                raise ValueError("RK3576 device smoke model identity/SHA mismatch")
            seen.add(stage)
            runs = model.get("runs")
            if not isinstance(runs, list) or [run.get("T") for run in runs if isinstance(run, dict)] != profiles:
                raise ValueError("RK3576 device smoke profile runs mismatch")
            for run in runs:
                t = run.get("T")
                row = expected[stage]
                required_sha = row.get("rknn_sha256", row.get("sha256"))
                if stage == "noise0":
                    static = row.get("static_profiles", {})
                    spec = static.get(f"T{t}") if isinstance(static, dict) else None
                    required_sha = spec.get("rknn_sha256") if isinstance(spec, dict) else None
                if run.get("rknn_sha256", model.get("rknn_sha256")) != required_sha:
                    raise ValueError("RK3576 device smoke model identity/SHA mismatch")
                f, l = 60 * int(t) + 1, 10 * int(t)
                if stage == "post": expected_shape = [1, 22, f]
                elif stage == "main1" or stage.startswith(("rb3-", "rb4-", "rb5-", "noise1-")): expected_shape = [1, 128, f]
                else: expected_shape = [1, 256, l]
                if run.get("output_shapes") != [expected_shape]:
                    raise ValueError(f"RK3576 device smoke output ABI mismatch: {stage} T{t}")
                if (run.get("load_return"), run.get("init_return"), run.get("finite"), run.get("release_success")) != (0, 0, True, True):
                    raise ValueError("RK3576 device smoke run gate failed")
                if not isinstance(run.get("inference_s"), (int, float)) or not np.isfinite(run["inference_s"]) or run["inference_s"] < 0:
                    raise ValueError("RK3576 device smoke timing mismatch")
                if not isinstance(run.get("output_shapes"), list) or not run["output_shapes"]:
                    raise ValueError("RK3576 device smoke output ABI missing")
        if seen != set(expected):
            raise ValueError("RK3576 device smoke stage set mismatch")
    def preload(self):
        with self._lifecycle_lock:
            if self.ready:
                return
            if self.frontend is None: raise RuntimeError("KokoroLong32Runtime requires an official frontend producing x/s/har; refusing legacy token frontend")
            if self.lite_cls is None:
                from rknnlite.api import RKNNLite
                self.lite_cls=RKNNLite
            self._manifest(); self._load_all()
            try:
                self.pool=ThreadPoolExecutor(max_workers=4, thread_name_prefix="kokoro-long32")
                self.ready=True
            except Exception as pool_error:
                try:
                    self.cleanup()
                except Exception as cleanup_error:
                    raise RuntimeError(
                        f"long32 executor construction failed: {type(pool_error).__name__}: {pool_error}; "
                        f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    ) from pool_error
                raise
            if not self._cleanup_registered:
                atexit.register(self.cleanup)
                self._cleanup_registered=True
    def _load_all(self):
        if self.contexts or self.branches or self.noise is not None:
            raise RuntimeError("previous long32 load retains resources; cleanup required before preload")
        C=self.lite_cls; root=self.root
        shapes={"main0":([(1,512,None)],(1,256,None)),"noise0":([(1,128),(1,22,None)],(1,256,None)),"main1":([(1,256,None)],(1,128,None)),"post":([(1,128,None)],(1,22,None))}
        for i in range(3):shapes[f"rb{i}"]=([(1,256,None),(1,128)],(1,256,None))
        owned_contexts = {}
        owned_branches = {}
        owned_noise = None
        try:
            base = ("main0", "rb0", "rb1", "rb2", "main1") + (() if self.prefix_only else ("post",))
            for n in base:
                try:
                    owned_contexts[n] = _Context(C(), root/"models"/f"{n}.fp16.rknn", *shapes[n], n, self.context_core_masks[n])
                except Exception as exc:
                    partial = getattr(exc, "unreleased_context", None)
                    if partial is not None: owned_contexts[n] = partial
                    raise
            if self.platform != "rk3576":
                try:
                    owned_contexts["noise0"] = _Context(C(), root/"models"/"noise0.fp16.rknn", *shapes["noise0"], "noise0", self.context_core_masks["noise0"])
                except Exception as exc:
                    partial = getattr(exc, "unreleased_context", None)
                    if partial is not None: owned_contexts["noise0"] = partial
                    raise
            for b in RB:
                try:
                    owned_branches[b] = _Branch(C, root, b, self.context_core_masks, load_contexts=not self.prefix_only)
                except Exception as exc:
                    partial = getattr(exc, "unreleased_component", None)
                    if partial is not None: owned_branches[b] = partial
                    raise
            try:
                owned_noise = _Noise1(C, root, self.context_core_masks)
            except Exception as exc:
                partial = getattr(exc, "unreleased_component", None)
                if partial is not None: owned_noise = partial
                raise
            self.contexts = owned_contexts
            self.branches = owned_branches
            self.noise = owned_noise
        except Exception as load_error:
            cleanup_errors = []
            retained_branches = {}
            for key, branch in reversed(list(owned_branches.items())):
                try: branch.release()
                except Exception as exc:
                    cleanup_errors.append(f"branch rb{key}: {exc}"); retained_branches[key] = branch
            retained_noise = None
            if owned_noise is not None:
                try: owned_noise.release()
                except Exception as exc:
                    cleanup_errors.append(f"noise1: {exc}"); retained_noise = owned_noise
            retained_contexts = {}
            for key, context in reversed(list(owned_contexts.items())):
                try: context.release()
                except Exception as exc:
                    cleanup_errors.append(f"context {key}: {exc}"); retained_contexts[key] = context
            self.contexts = retained_contexts
            self.branches = retained_branches
            self.noise = retained_noise
            self.ready = False
            if cleanup_errors:
                raise RuntimeError(
                    f"long32 load failed: {type(load_error).__name__}: {load_error}; "
                    "rollback cleanup errors: " + "; ".join(cleanup_errors)
                ) from load_error
            raise
    def run_profile(self, tensors: Mapping[str,Any], *, target_d: int | None = None, max_rtf: float | None = None, cancel_event: object | None = None):
        if self.prefix_only:
            raise RuntimeError("prefix-only runtime cannot run full generator")
        if not self.ready: raise RuntimeError("preload() required")
        x=np.asarray(tensors["x"]); t=int(x.shape[-1]);
        if t not in self.route_ts: raise ValueError("T is not enabled by KOKORO_LONG32_ROUTE_TS")
        f,l=60*t+1,10*t; x=_finite(x,(1,512,t),"x"); s=_finite(tensors["s"],(1,128),"s"); har=_finite(tensors["har"],(1,22,f),"har"); c=self.contexts; started=time.perf_counter();
        pool=self.pool
        if pool is None: raise RuntimeError("long32 worker pool is unavailable")
        if max_rtf is None: max_rtf=float(os.environ.get("KOKORO_LONG32_MAX_GENERATOR_RTF", "0.30"))
        if max_rtf <= 0 or not np.isfinite(max_rtf): raise ValueError("max_rtf must be finite and > 0")
        if _cancelled(cancel_event): raise Long32Cancelled()
        noise0_ctx = self._noise0_for(t)
        submitted = []

        def submit(fn, *args):
            future = pool.submit(fn, *args)
            submitted.append(future)
            return future

        def drain_futures() -> None:
            # cancel() only affects work that has not started.  A running NPU
            # call must be allowed to return before the request lock is
            # released; consuming every result also prevents late worker
            # exceptions from leaking into the next request.
            for future in submitted:
                future.cancel()
            for future in submitted:
                try:
                    future.result()
                except BaseException:
                    pass

        try:
            main_f, noise_f = submit(c["main0"].run,[x]), submit(noise0_ctx.run,[s,har]); noise1_f=submit(self.noise.run,har,s)
            main0=main_f.result()
            if _cancelled(cancel_event): raise Long32Cancelled()
            noise0=noise_f.result(); j0=main0+noise0
            if _cancelled(cancel_event): raise Long32Cancelled()
            rb_f=[submit(c[f"rb{i}"].run,[j0,s]) for i in range(3)]
            rb=[]
            for future in rb_f:
                rb.append(future.result())
                if _cancelled(cancel_event): raise Long32Cancelled()
            main1=c["main1"].run([(rb[0]+rb[1]+rb[2])/np.float32(3)])
            if _cancelled(cancel_event): raise Long32Cancelled()
            j1=main1+noise1_f.result()
            if _cancelled(cancel_event): raise Long32Cancelled()
            hi_f=[submit(self.branches[b].run,j1,s) for b in RB]
            hi=[]
            for future in hi_f:
                hi.append(future.result()[0])
                if _cancelled(cancel_event): raise Long32Cancelled()
            conv=c["post"].run([(hi[0]+hi[1]+hi[2])/np.float32(3)])
            if _cancelled(cancel_event): raise Long32Cancelled()
            wall_ms=(time.perf_counter()-started)*1000; duration=(5*(f-1))/24000; generator_rtf=wall_ms/1000/duration
            rtf_pass=bool(np.isfinite(generator_rtf) and generator_rtf <= max_rtf)
            if not rtf_pass and os.environ.get("KOKORO_LONG32_ENFORCE_RTF", "0") == "1":
                raise RuntimeError(f"long32 generator RTF gate failed: {generator_rtf:.4f} > {max_rtf}")
            return conv,{"wall_ms":wall_ms,"generator_rtf":generator_rtf,"generator_rtf_limit":max_rtf,"generator_rtf_pass":rtf_pass,"T":t,"F":f,"L":l,"duration_s":duration}
        finally:
            drain_futures()

    def _noise0_for(self, t: int) -> _Context:
        """Return the exact RK3576 static noise0 context for T.

        Loading is serialized and the cache is published only after load and
        init succeed, so concurrent first requests cannot observe a partial
        context.  RK3576 retains only the active profile: release the previous
        profile before loading a different one so two large static contexts and
        their runtime workspaces never overlap.  There is intentionally no
        nearest-profile fallback.
        """
        if self.platform != "rk3576":
            return self.contexts["noise0"]
        with self._noise0_lock:
            found = self.noise0_profiles.get(t)
            if found is not None:
                return found
            spec = self._noise0_specs.get(t)
            if spec is None:
                raise ValueError(f"RK3576 noise0 static profile T{t} is not declared")
            self._release_noise0_profiles_locked()
            ctx = _Context(self.lite_cls(), spec[0], [(1,128),(1,22,None)], (1,256,None), f"noise0-T{t}", self.context_core_masks["noise0"])
            self.noise0_profiles[t] = ctx
            return ctx

    def _release_noise0_profiles_locked(self) -> None:
        errors = []
        for profile_t, context in list(self.noise0_profiles.items())[::-1]:
            try:
                context.release()
            except Exception as exc:
                errors.append(f"T{profile_t}: {exc}")
            else:
                del self.noise0_profiles[profile_t]
        if errors:
            raise RuntimeError("noise0 profile release: " + "; ".join(errors))

    def release_profile_contexts(self) -> None:
        """Release lazy RK3576 profile contexts after serialized use."""
        with self._lifecycle_lock:
            with self._noise0_lock:
                self._release_noise0_profiles_locked()
    def _sentence_groups(self, text: str, *, language: str | None, speed: float, cancel_event: object | None = None) -> list[str]:
        """Group text by natural frontend duration, then split oversized units.

        Punctuation and whitespace are preferred boundaries.  Character binary
        search is the final lossless fallback for languages without spaces or
        punctuation.  Every returned slice is taken from the normalized input,
        so order and the trailing content are preserved.
        """
        source = str(text).strip()
        if not source or not hasattr(self.frontend, "prepare"):
            return [source or text]
        max_d = max(self.route_ts) // 2

        def natural(value: str) -> int:
            if _cancelled(cancel_event):
                raise Long32Cancelled()
            try:
                return int(self.frontend.prepare(value, language or "en-US", speed).natural_D)
            except Long32Cancelled:
                raise
            except Exception as exc:
                raise Long32Fallback(f"frontend prepare failed while splitting: {type(exc).__name__}: {exc}") from exc

        def split_unit(value: str) -> list[str]:
            if _cancelled(cancel_event):
                raise Long32Cancelled()
            if natural(value) <= max_d:
                return [value]
            # Prefer the largest punctuation boundary that fits, then spaces.
            boundaries: list[tuple[int, int]] = []
            for match in re.finditer(r"(?<=[.!?;。！？；])\s*", value):
                if match.end() < len(value): boundaries.append((match.end(), 0))
            for match in re.finditer(r"\s+", value):
                if match.end() < len(value): boundaries.append((match.end(), 1))
            for end, _priority in sorted(boundaries, key=lambda item: (-item[0], item[1])):
                if natural(value[:end]) <= max_d:
                    return split_unit(value[:end]) + split_unit(value[end:])
            # No semantic boundary fits: find the largest character prefix
            # accepted by the frontend's natural duration probe.
            lo, hi, best = 1, len(value) - 1, 0
            while lo <= hi:
                if _cancelled(cancel_event): raise Long32Cancelled()
                mid = (lo + hi) // 2
                if natural(value[:mid]) <= max_d:
                    best = mid; lo = mid + 1
                else:
                    hi = mid - 1
            if best <= 0:
                raise Long32Fallback("frontend natural duration cannot fit a character in the enabled route")
            return split_unit(value[:best]) + split_unit(value[best:])

        units = [m.group(0) for m in re.finditer(r".*?(?:(?<=[.!?;。！？；])\s*|$)", source, re.S) if m.group(0)]
        # The regex above can produce a final empty-ish match; split each
        # punctuation unit recursively before duration-based accumulation.
        units = [part for unit in units for part in split_unit(unit) if part]
        groups: list[str] = []; current = ""
        for unit in units:
            if _cancelled(cancel_event): raise Long32Cancelled()
            candidate = current + unit
            if current and natural(candidate) > max_d:
                groups.append(current); current = unit
            else:
                current = candidate
        if current: groups.append(current)
        return groups or [source]

    def _synthesize_one(self, text: str, *, speed: float, language: str | None, cancel_event: object | None = None, **kwargs):
        """Run an injected official frontend followed by RKNN and iSTFT."""
        if self.frontend is None: raise RuntimeError("official long32 frontend is not configured")
        started=time.perf_counter()
        profile = None
        if hasattr(self.frontend, "prepare") and hasattr(self.frontend, "render"):
            try:
                prepared = self.frontend.prepare(text, language or "en-US", speed)
                profile = select_profile(prepared.natural_D, max_ratio=parse_max_snap_ratio(os.environ.get("KOKORO_LONG32_MAX_SNAP_RATIO")), platform=self.platform, profiles=self.route_ts)
            except Exception as exc:
                raise Long32Fallback(f"frontend prepare failed: {type(exc).__name__}: {exc}") from exc
            if profile["fallback"]:
                failure = Long32Fallback(profile["reason"])
                failure.profile = profile
                raise failure
            try: tensors = self.frontend.render(prepared, profile["target_D"])
            except Exception as exc: raise Long32Fallback(f"frontend render failed: {type(exc).__name__}: {exc}") from exc
        else:
            tensors = self.frontend(text=text, speed=speed, language=language, **kwargs)
        if not isinstance(tensors, Mapping): raise TypeError("long32 frontend must return a tensor mapping")
        if _cancelled(cancel_event): raise Long32Cancelled()
        conv, meta = self.run_profile(tensors, cancel_event=cancel_event)
        audio = _istft(conv)[0]
        total_ms=(time.perf_counter()-started)*1000; total_rtf=total_ms/1000/max(audio.size/24000,1e-9)
        return audio, {**meta, "backend": "kokoro_long32", "language": language,
                       "route_ts": list(self.route_ts), "selected_T": meta.get("T"),
                       "snap_ratio": profile["snap_ratio"] if profile is not None else None, "total_wall_ms": total_ms,
                       "total_rtf": total_rtf, "fallback": False}

    def synthesize(self, text: str, *, speed: float = 1.0, language: str | None = None, segments: Any = None, cancel_event: object | None = None, **kwargs):
        with self._lifecycle_lock:
            if self.prefix_only:
                raise RuntimeError("prefix-only runtime cannot synthesize full generator")
            return self._synthesize_locked(text, speed=speed, language=language, segments=segments, cancel_event=cancel_event, **kwargs)

    def _synthesize_locked(self, text: str, *, speed: float = 1.0, language: str | None = None, segments: Any = None, cancel_event: object | None = None, **kwargs):
        request_started=time.perf_counter(); chunks=[]; metas=[]
        fallback = kwargs.pop("fallback", None)
        if segments is not None:
            groups = list(segments)
            grouping_error = None
        else:
            grouping_error = None
            try:
                groups = self._sentence_groups(text, language=language, speed=speed, cancel_event=cancel_event)
            except Long32Cancelled:
                total_ms = (time.perf_counter() - request_started) * 1000
                return np.zeros(0, np.float32), {
                    "backend": "kokoro_long32", "duration_s": 0.0, "duration": 0.0,
                    "wall_ms": total_ms, "inference_time": total_ms / 1000.0,
                    "rtf": 0.0, "total_rtf": 0.0, "route_ts": list(self.route_ts),
                    "T": [], "selected_T": [], "snap_ratio": [], "fallback": False,
                    "cancelled": True,
                }
            except Long32Fallback as exc:
                if fallback is None:
                    raise
                groups = [text]
                grouping_error = exc
        for group in groups:
            if _cancelled(cancel_event): break
            if grouping_error is not None:
                exc = grouping_error
                grouping_error = None
                fallback_started = time.perf_counter()
                try:
                    audio, meta = fallback(group, speed=speed, language=language, reason=str(exc), cancel_event=cancel_event)
                except TypeError as type_exc:
                    if "cancel_event" not in str(type_exc): raise
                    audio, meta = fallback(group, speed=speed, language=language, reason=str(exc))
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                fallback_ms = (time.perf_counter() - fallback_started) * 1000
                meta = {**meta, "engine": "legacy_hybrid", "fallback": True,
                        "fallback_reason": str(exc), "route_ts": list(self.route_ts),
                        "selected_T": None, "snap_ratio": None,
                        "total_wall_ms": fallback_ms}
                chunks.append(audio); metas.append(meta)
                continue
            try: audio, meta = self._synthesize_one(group, speed=speed, language=language, cancel_event=cancel_event, **kwargs)
            except Long32Cancelled:
                break
            except Long32Fallback as exc:
                if fallback is None: raise
                if _cancelled(cancel_event): break
                fallback_started=time.perf_counter()
                try: audio, meta = fallback(group, speed=speed, language=language, reason=str(exc), cancel_event=cancel_event)
                except TypeError as type_exc:
                    if "cancel_event" not in str(type_exc): raise
                    audio, meta = fallback(group, speed=speed, language=language, reason=str(exc))
                audio=np.asarray(audio, dtype=np.float32).reshape(-1)
                fallback_ms=(time.perf_counter()-fallback_started)*1000
                profile = getattr(exc, "profile", {})
                meta = {**meta, "engine":"legacy_hybrid", "fallback":True, "fallback_reason":str(exc), "route_ts":list(self.route_ts), "selected_T":profile.get("T"), "snap_ratio":profile.get("snap_ratio"), "total_wall_ms":fallback_ms}
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1)); metas.append(meta)
        audio=np.concatenate(chunks) if chunks else np.zeros(0,np.float32)
        total_ms=(time.perf_counter()-request_started)*1000; duration=audio.size/24000
        total_rtf=total_ms/1000/max(duration,1e-9)
        long_parts=[m for m in metas if not m.get("fallback")]
        fallback_parts=[m for m in metas if m.get("fallback")]
        long_duration=sum(float(m.get("duration_s", 0.0)) for m in long_parts)
        long_wall=sum(float(m.get("wall_ms", m.get("total_wall_ms", 0.0))) for m in long_parts)
        fallback_duration=sum(float(m.get("duration_s", 0.0)) for m in fallback_parts)
        fallback_wall=sum(float(m.get("total_wall_ms", 0.0)) for m in fallback_parts)
        return audio, {"backend":"kokoro_long32", "duration_s":duration, "duration":duration, "wall_ms":total_ms,
                       "inference_time":total_ms / 1000.0, "rtf":total_rtf,
                       "generator_rtf": (long_wall/1000/max(long_duration,1e-9)) if long_parts else None,
                       "fallback_rtf": (fallback_wall/1000/max(fallback_duration,1e-9)) if fallback_parts else None,
                       "total_rtf":total_rtf, "T":[m.get("T") for m in metas],
                       "selected_T":[m.get("selected_T", m.get("T")) for m in metas],
                       "route_ts":list(self.route_ts),
                       "snap_ratio":[m.get("snap_ratio") for m in metas],
                       "fallback_reasons":[m.get("fallback_reason") for m in metas if m.get("fallback_reason")],
                       "fallback":bool(fallback_parts), "mixed":bool(long_parts and fallback_parts),
                       "fallback_count":len(fallback_parts), "segments":len(metas), "segment_meta":metas,
                       "cancelled": _cancelled(cancel_event)}
    def cleanup(self):
        with self._lifecycle_lock:
            self._cleanup_locked()

    def _cleanup_locked(self):
        errors=[]
        pool=self.pool
        if pool is not None:
            try: pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:errors.append(str(e))
            else: self.pool=None
        for key, b in list(self.branches.items())[::-1]:
            try: b.release()
            except Exception as e: errors.append(f"branch rb{key}: {e}")
            else: del self.branches[key]
        if self.noise is not None:
            try: self.noise.release()
            except Exception as e: errors.append(f"noise1: {e}")
            else: self.noise=None
        for key, c in list(self.contexts.items())[::-1]:
            try: c.release()
            except Exception as e: errors.append(f"context {key}: {e}")
            else: del self.contexts[key]
        with self._noise0_lock:
            try: self._release_noise0_profiles_locked()
            except Exception as e: errors.append(str(e))
        self.ready=False
        if errors: raise RuntimeError("long32 cleanup errors: "+"; ".join(errors))
    def runtime_info(self):
        symbols = {"AUTO": 0, "CORE0": 1, "CORE1": 2, "CORE2": 4}
        active = {key: value for key, value in self.context_core_masks.items() if value}
        loaded = len(self.contexts) + sum(len(branch.ctx) for branch in self.branches.values())
        loaded += len(self.noise.ctx) if self.noise is not None else 0
        loaded += len(self.noise0_profiles)
        return {"backend":"kokoro_long32", "ready":self.ready, "platform":self.platform,
                "logical_contexts":32, "contexts":32, "profiles":list(PLATFORM_TS[self.platform]),
                "route_ts":list(self.route_ts), "prefix_only":self.prefix_only, "core_mask_symbols":symbols,
                "context_core_masks":dict(self.context_core_masks),
                "active_core_masks":active, "loaded_physical_contexts":loaded}


KokoroLong32Backend = KokoroLong32Runtime
