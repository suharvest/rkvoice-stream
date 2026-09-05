import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from rkvoice_stream.backends.tts.kokoro_long32 import CONTEXT_STAGES, Long32Cancelled, Long32Fallback, KokoroLong32Runtime, _Context, parse_max_snap_ratio, parse_route_ts, select_profile, snap_durations


def _bundle(tmp_path: Path, **overrides):
    models = tmp_path / "models"; models.mkdir(parents=True)
    rows = []
    target = overrides.get("target", "rk3588")
    profile_ts = (320, 368, 416, 480, 576, 640) if target == "rk3576" else (320,400,480,560,640)
    for i, stage in enumerate(CONTEXT_STAGES):
        if target == "rk3576" and stage == "noise0":
            static = {}
            for t in profile_ts:
                name = f"noise0-T{t}.fp16.rknn"; path = models / name; path.write_bytes(f"noise0-{t}".encode())
                static[f"T{t}"] = {"T": t, "F": 60*t+1, "L": 10*t, "rknn": name, "rknn_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "inputs": [[1,128],[1,22,60*t+1]], "outputs": [[1,256,10*t]], "input_count": 2, "output_count": 1}
            rows.append({"stage": stage, "status": "PASS", "static_profiles": static})
            continue
        name = f"{stage}.fp16.rknn"; path = models / name; path.write_bytes(f"{i}".encode())
        rows.append({"stage": stage, "status": "PASS", "rknn": name,
                     "rknn_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    export_path = tmp_path / "generator-export.manifest.json"
    export_path.write_text(json.dumps({"schema": "kokoro-v10-generator-export-1", "status": "PASS", "target": "rk3588", "full_model_sha256": "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb", "output_sha256": "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c"}))
    export_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
    obj = {"schema": f"kokoro-v10-{target}-long-hybrid-full-v6", "status": "PASS", "full_source_sha256": "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb", "generator_source_sha256": "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c", "source_sha256": "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c", "source_sha256_deprecated": True, "source_sha256_kind": "generator_model", "generator_export_manifest_sha256": export_sha, "preflight_evidence": {"generator_export_manifest_sha256": export_sha}, "toolkit": "2.3.2", "target": target, "precision": "FP16",
           "optimization_level": 3, "contexts": 32, "rows": rows,
           "profiles": {f"T{t}": {"T": t, "L": 10*t, "F": 60*t+1} for t in profile_ts}}
    obj.update(overrides)
    obj["schema"] = f"kokoro-v10-{target}-long-hybrid-full-v7"
    (tmp_path / "build-manifest.json").write_text(json.dumps(obj))
    return tmp_path


def _rk3576_smoke(root: Path):
    manifest_path = root / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profiles = [320, 368, 416, 480, 576, 640]
    report = {
        "schema": "kokoro-v10-isolated-smoke-2",
        "status": "PASS",
        "target": "rk3576",
        "build_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "full_source_sha256": manifest["full_source_sha256"], "generator_source_sha256": manifest["generator_source_sha256"], "generator_export_manifest_sha256": manifest["generator_export_manifest_sha256"],
        "required_runs": 192,
        "passed_runs": 192,
        "profiles": profiles,
        "models": [{
            "stage": row["stage"],
            "rknn_sha256": row.get("rknn_sha256"),
            "runs": [{"T": t, "load_return": 0, "init_return": 0,
                      "rknn_sha256": (row["static_profiles"][f"T{t}"]["rknn_sha256"] if row["stage"] == "noise0" else row["rknn_sha256"]),
                      "inference_s": 0.01, "output_shapes": [[1, 22, 60*t+1] if row["stage"] == "post" else ([1, 128, 60*t+1] if row["stage"] == "main1" or row["stage"].startswith(("rb3-", "rb4-", "rb5-", "noise1-")) else [1, 256, 10*t])],
                      "finite": True, "release_success": True} for t in profiles],
        } for row in manifest["rows"]],
    }
    (root / "device-smoke-report.json").write_text(json.dumps(report))
    return report


def test_profile_selection_and_snap_rules():
    assert select_profile(160)["T"] == 320
    assert select_profile(200)["T"] == 400
    assert select_profile(240)["T"] == 480
    assert select_profile(280)["T"] == 560
    assert select_profile(320)["T"] == 640
    assert select_profile(20)["fallback"] is True
    assert np.array_equal(snap_durations([3, 4, 5], 12), np.array([3, 4, 5]))
    snapped = snap_durations([3, 4, 5], 15)
    assert snapped.sum() == 15 and np.all(snapped >= 1)


@pytest.mark.parametrize("raw", ["640,400", "400,400", "400, 640", ",400", "400,", "bogus", "400,999"])
def test_route_ts_parser_is_strict_and_platform_bound(raw):
    with pytest.raises(ValueError):
        parse_route_ts(raw, platform="rk3588")


def test_route_ts_parser_defaults_to_all_and_select_profile_uses_only_route():
    assert parse_route_ts(None, platform="rk3588") == (320, 400, 480, 560, 640)
    assert parse_route_ts("", platform="rk3576") == (320, 368, 416, 480, 576, 640)
    assert [select_profile(d, platform="rk3588", profiles=(400, 640))["T"] for d in (160, 200, 240, 280, 320)] == [400, 400, 400, 640, 640]


@pytest.mark.parametrize("raw", ["", "nan", "inf", "-inf", "-0.1", "1.01", "2", "bogus"])
def test_max_snap_ratio_parser_fails_closed(raw):
    with pytest.raises(ValueError):
        parse_max_snap_ratio(raw)


def test_max_snap_ratio_parser_accepts_default_and_finite_bounds():
    assert parse_max_snap_ratio() == pytest.approx(0.10)
    assert parse_max_snap_ratio("0") == 0.0
    assert parse_max_snap_ratio("1") == 1.0


def test_rk3576_profiles_cover_four_to_eight_seconds_within_ten_percent():
    expected = (320, 368, 416, 480, 576, 640)
    assert [select_profile(d, platform="rk3576")["T"] for d in (160, 184, 208, 240, 288, 320)] == list(expected)
    for natural_d in range(160, 321):
        assert select_profile(natural_d, platform="rk3576")["fallback"] is False


def test_rk3576_manifest_requires_hash_bound_device_smoke(tmp_path):
    profiles = {f"T{t}": {"T": t, "L": 10*t, "F": 60*t+1} for t in (320,368,416,480,576,640)}
    root = _bundle(tmp_path, target="rk3576", profiles=profiles)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, platform="rk3576")
    with pytest.raises(ValueError, match="smoke report missing"):
        runtime._manifest()
    report = _rk3576_smoke(root)
    runtime._manifest()
    report["build_manifest_sha256"] = "0" * 64
    (root / "device-smoke-report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="build-manifest SHA"):
        runtime._manifest()


def test_manifest_context_target_and_sha_fail_closed(tmp_path):
    root = _bundle(tmp_path)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {})
    runtime._manifest()
    bad = json.loads((root / "build-manifest.json").read_text()); bad["contexts"] = 31
    (root / "build-manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="context"):
        runtime._manifest()
    bad["contexts"] = 32; bad["rows"] = bad["rows"][:-1]
    (root / "build-manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="rows"):
        runtime._manifest()
    bad["rows"] = json.loads(_bundle(tmp_path / "other").joinpath("build-manifest.json").read_text())["rows"]
    bad["rows"][0]["rknn_sha256"] = "0" * 64
    (root / "build-manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="SHA"):
        runtime._manifest()


@pytest.mark.parametrize("field", ["full_source_sha256", "generator_source_sha256", "generator_export_manifest_sha256"])
def test_manifest_dual_provenance_tamper_fails_closed(tmp_path, field):
    root = _bundle(tmp_path)
    path = root / "build-manifest.json"; data = json.loads(path.read_text()); data[field] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dual source|generator export"):
        KokoroLong32Runtime(root, frontend=lambda **_: {})._manifest()


def test_manifest_preflight_chain_mismatch_fails_closed(tmp_path):
    root = _bundle(tmp_path)
    path = root / "build-manifest.json"; data = json.loads(path.read_text())
    data["preflight_evidence"]["generator_export_manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="preflight generator export"):
        KokoroLong32Runtime(root, frontend=lambda **_: {})._manifest()


def test_manifest_requires_exact_stage_set(tmp_path):
    root = _bundle(tmp_path)
    bad = json.loads((root / "build-manifest.json").read_text())
    bad["rows"][0]["stage"] = bad["rows"][1]["stage"]
    (root / "build-manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="stage set"):
        KokoroLong32Runtime(root, frontend=lambda **_: {})._manifest()


def test_frontend_is_required_and_release_is_idempotent(tmp_path):
    root = _bundle(tmp_path)
    runtime = KokoroLong32Runtime(root)
    with pytest.raises(RuntimeError, match="frontend"):
        runtime.preload()
    runtime.cleanup(); runtime.cleanup()


def test_cleanup_waits_for_active_synthesize_lifecycle():
    entered = threading.Event(); unblock = threading.Event(); cleaned = threading.Event()
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    def work(*args, **kwargs):
        entered.set(); assert unblock.wait(2); return np.zeros(0, np.float32), {"fallback": False}
    runtime._synthesize_locked = work
    worker = threading.Thread(target=lambda: runtime.synthesize("x")); worker.start()
    assert entered.wait(2)
    cleaner = threading.Thread(target=lambda: (runtime.cleanup(), cleaned.set())); cleaner.start()
    assert not cleaned.wait(0.05)
    unblock.set(); worker.join(2); cleaner.join(2)
    assert cleaned.is_set()


def test_release_all_32_contexts(tmp_path):
    class Lite:
        released = 0
        def load_rknn(self, _): return 0
        def init_runtime(self, **_): return 0
        def release(self): Lite.released += 1
    contexts = []
    for i in range(32):
        p = tmp_path / f"long32-{i}" / "model.rknn"
        p.parent.mkdir()
        p.write_bytes(b"rknn")
        contexts.append(_Context(Lite(), p, [(1, 1)], (1, 1), str(i)))
    for context in reversed(contexts): context.release()
    assert Lite.released == 32


def test_preload_loads_exact_7_18_7_contexts_with_bundle_names(tmp_path):
    names = ["main0.fp16.rknn", "noise0.fp16.rknn", "rb0.fp16.rknn", "rb1.fp16.rknn", "rb2.fp16.rknn", "main1.fp16.rknn", "post.fp16.rknn"]
    names += [f"rb{b}-convs{h}_{i}.fp16.rknn" for b in (3,4,5) for h in (1,2) for i in range(3)]
    names += ["noise1-preconv.fp16.rknn"] + [f"noise1-convs{h}_{i}.fp16.rknn" for h in (1,2) for i in range(3)]
    root = tmp_path; (root / "models").mkdir()
    rows=[]
    for name in names:
        p=root/"models"/name; p.write_bytes(name.encode()); rows.append({"stage":name.removesuffix(".fp16.rknn"),"status":"PASS","rknn":name,"rknn_sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    obj=json.loads(((_bundle(tmp_path / "meta") / "build-manifest.json").read_text()))
    (root / "generator-export.manifest.json").write_bytes((tmp_path / "meta" / "generator-export.manifest.json").read_bytes())
    obj["rows"]=rows; (root/"build-manifest.json").write_text(json.dumps(obj))
    for group, stems in (("rb345", [f"rb{b}-adain{h}_{i}" for b in (3,4,5) for h in (1,2) for i in range(3)]), ("noise1", [f"noise1-adain{h}_{i}" for h in (1,2) for i in range(3)])):
        (root/"params"/group).mkdir(parents=True)
        for stem in stems:
            np.zeros(256*128, np.float32).tofile(root/"params"/group/(stem+"-fc-weight.bin")); np.zeros(256,np.float32).tofile(root/"params"/group/(stem+"-fc-bias.bin")); np.ones(128,np.float32).tofile(root/"params"/group/(stem+"-alpha.bin"))
    class Lite:
        created=0; released=0
        def __init__(self): Lite.created += 1
        def load_rknn(self, _): return 0
        def init_runtime(self, **_): return 0
        def release(self): Lite.released += 1
    runtime=KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=Lite)
    runtime.preload()
    assert Lite.created == 32 and len(runtime.contexts) == 7 and len(runtime.branches) == 3 and len(runtime.noise.ctx) == 7
    runtime.cleanup(); assert Lite.released == 32


def test_duration_driven_grouping_accumulates_short_sentences():
    class Frontend:
        def prepare(self, text, *_):
            return type("Prepared", (), {"natural_D": 60 * text.count("。")} )()
    runtime = KokoroLong32Runtime("/unused", frontend=Frontend())
    groups = runtime._sentence_groups("一。二。三。四。五。", language="zh", speed=1.0)
    assert groups == ["一。二。三。四。五。"]


def test_duration_grouping_splits_unpunctuated_text_without_tail_loss(monkeypatch):
    class Frontend:
        def prepare(self, text, *_):
            return type("Prepared", (), {"natural_D": len(text) * 10})()
    monkeypatch.setenv("KOKORO_LONG32_ROUTE_TS", "400,640")
    runtime = KokoroLong32Runtime("/unused", frontend=Frontend())
    source = "无标点文本" * 100
    groups = runtime._sentence_groups(source, language="zh", speed=1.0)
    assert "".join(groups) == source
    assert all(len(group) <= 32 for group in groups)


def test_fallback_callback_returns_float_audio(monkeypatch):
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    monkeypatch.setattr(runtime, "_sentence_groups", lambda *a, **k: ["短句"])
    monkeypatch.setattr(runtime, "_synthesize_one", lambda *a, **k: (_ for _ in ()).throw(Long32Fallback("short")))
    audio, meta = runtime.synthesize(
        "短句", language="zh",
        fallback=lambda *a, **k: (np.zeros(2400, np.float32), {"total_wall_ms": 1.0}),
    )
    assert audio.dtype == np.float32 and audio.shape == (2400,)
    assert meta["fallback"] is True and meta["fallback_count"] == 1


def test_istft_runs_once_and_total_includes_frontend(monkeypatch):
    class Frontend:
        def prepare(self, *_): return type("Prepared", (), {"natural_D": 160})()
        def render(self, *_): return {"x": np.zeros((1,512,320),np.float32), "s": np.zeros((1,128),np.float32), "har": np.zeros((1,22,19201),np.float32)}
    runtime = KokoroLong32Runtime("/unused", frontend=Frontend())
    monkeypatch.setattr(runtime, "run_profile", lambda *_a, **_k: (np.zeros((1,22,19201),np.float32), {"wall_ms":0.0,"generator_rtf":0.001,"T":320,"F":19201,"L":3200,"duration_s":4.0}))
    calls = []
    monkeypatch.setattr("rkvoice_stream.backends.tts.kokoro_long32._istft", lambda conv: calls.append(conv) or np.zeros((1,96000),np.float32))
    audio, meta = runtime._synthesize_one("四秒句子", speed=1.0, language="zh")
    assert len(calls) == 1 and audio.shape == (96000,)
    assert meta["total_wall_ms"] >= meta["wall_ms"]


def test_noise1_release_is_reverse_and_raises():
    from rkvoice_stream.backends.tts.kokoro_long32 import _Noise1
    obj = object.__new__(_Noise1)
    events=[]
    class Context:
        def __init__(self, n): self.n=n; self.calls=0
        def release(self):
            self.calls += 1
            events.append(self.n)
            if self.n == 1 and self.calls == 1: raise RuntimeError("release-1")
    obj.ctx=[Context(i) for i in range(3)]
    with pytest.raises(RuntimeError, match="release-1"):
        obj.release()
    assert events == [2, 1, 0] and [context.n for context in obj.ctx] == [1]
    obj.release()
    assert events == [2, 1, 0, 1] and obj.ctx == []


def test_aggregate_rtf_does_not_use_fallback_as_zero(monkeypatch):
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    monkeypatch.setattr(runtime, "_sentence_groups", lambda *a, **k: ["long", "short"])
    monkeypatch.setattr(runtime, "_synthesize_one", lambda text, **_: (
        np.zeros(24_000, np.float32), {"wall_ms": 500.0, "generator_rtf": 0.5, "duration_s": 1.0, "fallback": False}
    ) if text == "long" else (_ for _ in ()).throw(Long32Fallback("short")))
    audio, meta = runtime.synthesize("x", fallback=lambda *a, **k: (
        np.zeros(24_000, np.float32), {"total_wall_ms": 2_000.0, "duration_s": 1.0}
    ))
    assert meta["generator_rtf"] == pytest.approx(0.5)
    assert meta["fallback_rtf"] > 0.0
    assert meta["mixed"] is True


def _complete_bundle(tmp_path, *, target="rk3588"):
    root = _bundle(tmp_path, target=target)
    for group, stems in (("rb345", [f"rb{b}-adain{h}_{i}" for b in (3, 4, 5) for h in (1, 2) for i in range(3)]),
                         ("noise1", [f"noise1-adain{h}_{i}" for h in (1, 2) for i in range(3)])):
        (root / "params" / group).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            np.zeros(256 * 128, np.float32).tofile(root / "params" / group / (stem + "-fc-weight.bin"))
            np.zeros(256, np.float32).tofile(root / "params" / group / (stem + "-fc-bias.bin"))
            np.ones(128, np.float32).tofile(root / "params" / group / (stem + "-alpha.bin"))
    if target == "rk3576":
        _rk3576_smoke(root)
    return root


@pytest.mark.parametrize("value", [None, [], 1, "x", 1.5])
def test_context_masks_require_mapping(tmp_path, value):
    if value is None:
        return
    with pytest.raises(TypeError, match="Mapping"):
        KokoroLong32Runtime(tmp_path, context_core_masks=value)


@pytest.mark.parametrize("mapping", [{"unknown": 1}, {"main0": True}, {"main0": "1"}, {"main0": 1.5}, {"main0": 3}])
def test_context_masks_reject_unknown_and_invalid_values(tmp_path, mapping):
    with pytest.raises((TypeError, ValueError)):
        KokoroLong32Runtime(tmp_path, context_core_masks=mapping)


def test_context_masks_are_copied_and_completed_with_auto(tmp_path):
    source = {"main0": 1, "noise1-convs1_0": 4}
    runtime = KokoroLong32Runtime(tmp_path, context_core_masks=source)
    source["main0"] = 2
    assert len(runtime.context_core_masks) == 32
    assert runtime.context_core_masks["main0"] == 1
    assert runtime.context_core_masks["noise1-convs1_0"] == 4
    assert runtime.context_core_masks["noise0"] == 0


class _MaskLite:
    created = []
    released = []
    init_calls = []
    no_kw = False

    def __init__(self):
        self.ident = len(type(self).created)
        type(self).created.append(self)

    def load_rknn(self, path):
        self.path = path
        return 0

    @property
    def NPU_CORE_AUTO(self):
        return 0

    def init_runtime(self, **kwargs):
        if self.no_kw and kwargs:
            raise TypeError("legacy signature")
        type(self).init_calls.append((Path(self.path).name, dict(kwargs)))
        return 0

    def release(self):
        type(self).released.append(self)


def _reset_mask_lite():
    _MaskLite.created = []; _MaskLite.released = []; _MaskLite.init_calls = []; _MaskLite.no_kw = False


def test_context_auto_and_legacy_signature_fallback(tmp_path):
    _reset_mask_lite(); path = tmp_path / "model.rknn"; path.write_bytes(b"x")
    lite = _MaskLite(); lite.no_kw = True
    ctx = _Context(lite, path, [(1, 1)], (1, 1), "auto")
    assert lite.init_calls == [("model.rknn", {})]  # legacy fallback call has no kwargs
    ctx.release()
    assert len(_MaskLite.released) == 1


@pytest.mark.parametrize("mask", [1, 2, 4])
def test_context_explicit_mask_and_typeerror_never_fallback(tmp_path, mask):
    _reset_mask_lite(); path = tmp_path / "model.rknn"; path.write_bytes(b"x")
    lite = _MaskLite(); lite.no_kw = True
    with pytest.raises(TypeError, match="legacy signature"):
        _Context(lite, path, [(1, 1)], (1, 1), "explicit", mask)
    assert len(_MaskLite.released) == 1


def test_context_release_failure_is_retryable(tmp_path):
    path = tmp_path / "model.rknn"; path.write_bytes(b"x")
    class Lite:
        calls = 0
        def load_rknn(self, _): return 0
        def init_runtime(self, **_): return 0
        def release(self):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("transient release")
    context = _Context(Lite(), path, [(1, 1)], (1, 1), "retry")
    with pytest.raises(RuntimeError, match="transient release"):
        context.release()
    assert context.released is False and context.obj.calls == 1
    context.release(); context.release()
    assert context.released is True and context.obj.calls == 2


def test_context_init_and_cleanup_failures_preserve_retry_owner(tmp_path):
    path = tmp_path / "model.rknn"; path.write_bytes(b"x")
    class Lite:
        def __init__(self): self.release_calls = 0
        def load_rknn(self, _): raise RuntimeError("load broke")
        def release(self):
            self.release_calls += 1
            if self.release_calls == 1: raise RuntimeError("cleanup broke")
    with pytest.raises(RuntimeError, match="load/init failed:.*load broke.*cleanup failed:.*cleanup broke") as caught:
        _Context(Lite(), path, [(1, 1)], (1, 1), "partial")
    context = caught.value.unreleased_context
    assert context.released is False
    context.release()
    assert context.released is True and context.obj.release_calls == 2


def test_preload_records_all_32_logical_masks(tmp_path):
    _reset_mask_lite(); root = _complete_bundle(tmp_path)
    mapping = {key: (1 if i % 3 == 0 else 2 if i % 3 == 1 else 4) for i, key in enumerate(CONTEXT_STAGES)}
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, context_core_masks=mapping)
    runtime.preload()
    calls = {Path(path).stem.removesuffix(".fp16"): kwargs.get("core_mask", 0) for (path, kwargs) in _MaskLite.init_calls}
    assert len(calls) == 32
    assert calls["main0"] == mapping["main0"]
    assert calls["rb3-convs1_0"] == mapping["rb3-convs1_0"]
    assert calls["noise1-convs2_2"] == mapping["noise1-convs2_2"]
    runtime.cleanup(); assert len(_MaskLite.released) == 32
    runtime.cleanup(); assert len(_MaskLite.released) == 32


@pytest.mark.parametrize("failure_name", ["main0.fp16.rknn", "rb3-convs1_0.fp16.rknn", "noise1-preconv.fp16.rknn"])
def test_preload_is_transactional_on_base_branch_noise_failure(tmp_path, failure_name):
    _reset_mask_lite(); root = _complete_bundle(tmp_path)
    class Failing(_MaskLite):
        def load_rknn(self, path):
            if Path(path).name == failure_name:
                raise RuntimeError("injected load failure")
            return super().load_rknn(path)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=Failing)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.preload()
    assert len(Failing.created) == len(Failing.released)
    assert runtime.contexts == {} and runtime.branches == {} and runtime.noise is None and not runtime.ready
    runtime.cleanup(); assert len(Failing.created) == len(Failing.released)


def test_load_failure_reports_rollback_failure_and_retains_for_cleanup(tmp_path):
    root = _complete_bundle(tmp_path)
    class FailingLite(_MaskLite):
        def __init__(self):
            super().__init__(); self.release_calls = 0
        def load_rknn(self, path):
            result = super().load_rknn(path)
            if Path(path).name == "main1.fp16.rknn":
                raise RuntimeError("injected main1 load")
            return result
        def release(self):
            self.release_calls += 1
            name = Path(self.path).name
            if name in {"main0.fp16.rknn", "rb0.fp16.rknn"} and self.release_calls == 1:
                raise RuntimeError(f"injected {name} cleanup")
            super().release()
    FailingLite.created = []; FailingLite.released = []; FailingLite.init_calls = []
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=FailingLite)
    with pytest.raises(RuntimeError, match="long32 load failed:.*main1.*rollback cleanup errors") as caught:
        runtime.preload()
    assert isinstance(caught.value.__cause__, RuntimeError)
    message = str(caught.value)
    assert "injected rb0.fp16.rknn cleanup" in message
    assert "injected main0.fp16.rknn cleanup" in message
    assert set(runtime.contexts) == {"main0", "rb0"}
    assert all(not context.released for context in runtime.contexts.values())
    with pytest.raises(RuntimeError, match="cleanup required"):
        runtime.preload()
    runtime.cleanup()
    assert runtime.contexts == {} and len(FailingLite.created) == len(FailingLite.released)


def test_preload_executor_failure_releases_contexts(tmp_path, monkeypatch):
    _reset_mask_lite(); root = _complete_bundle(tmp_path)
    class BrokenPool:
        def __init__(self, **kwargs): raise RuntimeError("executor failure")
    monkeypatch.setattr("rkvoice_stream.backends.tts.kokoro_long32.ThreadPoolExecutor", BrokenPool)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite)
    with pytest.raises(RuntimeError, match="executor failure"):
        runtime.preload()
    assert len(_MaskLite.created) == len(_MaskLite.released) == 32
    assert not runtime.ready and runtime.contexts == {} and runtime.branches == {} and runtime.noise is None


def test_executor_and_cleanup_failure_are_chained_and_retryable(tmp_path, monkeypatch):
    root = _complete_bundle(tmp_path)
    class ReleaseFailure(_MaskLite):
        def __init__(self): super().__init__(); self.release_calls = 0
        def release(self):
            self.release_calls += 1
            if Path(self.path).name == "main0.fp16.rknn" and self.release_calls == 1:
                raise RuntimeError("injected executor rollback cleanup")
            super().release()
    class BrokenPool:
        def __init__(self, **kwargs): raise RuntimeError("injected executor construction")
    ReleaseFailure.created = []; ReleaseFailure.released = []; ReleaseFailure.init_calls = []
    monkeypatch.setattr("rkvoice_stream.backends.tts.kokoro_long32.ThreadPoolExecutor", BrokenPool)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=ReleaseFailure)
    with pytest.raises(RuntimeError, match="executor construction failed:.*injected executor construction.*cleanup failed:.*injected executor rollback cleanup") as caught:
        runtime.preload()
    assert str(caught.value.__cause__) == "injected executor construction"
    assert set(runtime.contexts) == {"main0"} and runtime.pool is None and not runtime.ready
    assert not runtime.contexts["main0"].released
    runtime.cleanup(); runtime.cleanup()
    assert runtime.contexts == {}
    assert len(ReleaseFailure.created) == len(ReleaseFailure.released) == 32


def test_rk3576_lazy_noise0_inherits_logical_mask_and_cleanup(tmp_path):
    _reset_mask_lite(); root = _complete_bundle(tmp_path, target="rk3576")
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, platform="rk3576", context_core_masks={"noise0": 4})
    runtime.preload(); assert len(_MaskLite.created) == 31
    ctx = runtime._noise0_for(320)
    assert ctx.core_mask == 4
    assert _MaskLite.init_calls[-1][1]["core_mask"] == 4
    runtime.cleanup(); assert len(_MaskLite.created) == len(_MaskLite.released) == 32


def test_rk3576_keeps_only_active_noise0_profile_and_releases_before_load(tmp_path):
    """A profile switch cannot overlap two simulated 256 MiB workspaces."""
    root = _complete_bundle(tmp_path, target="rk3576")

    class SizedLite(_MaskLite):
        resident = peak = 0
        events = []

        def init_runtime(self, **kwargs):
            result = super().init_runtime(**kwargs)
            if Path(self.path).name.startswith("noise0-T"):
                type(self).resident += 256 * 1024 * 1024
                type(self).peak = max(type(self).peak, type(self).resident)
                type(self).events.append(("load", Path(self.path).name))
            return result

        def release(self):
            if Path(self.path).name.startswith("noise0-T"):
                type(self).resident -= 256 * 1024 * 1024
                type(self).events.append(("release", Path(self.path).name))
            super().release()

    SizedLite.created = []; SizedLite.released = []; SizedLite.init_calls = []
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=SizedLite,
                                  platform="rk3576")
    runtime.preload()
    first = runtime._noise0_for(320)
    assert runtime._noise0_for(320) is first
    second = runtime._noise0_for(640)
    assert second is not first and set(runtime.noise0_profiles) == {640}
    assert SizedLite.events == [("load", "noise0-T320.fp16.rknn"),
                                ("release", "noise0-T320.fp16.rknn"),
                                ("load", "noise0-T640.fp16.rknn")]
    assert SizedLite.peak == SizedLite.resident == 256 * 1024 * 1024
    runtime.release_profile_contexts()
    runtime.release_profile_contexts()
    assert SizedLite.resident == 0 and runtime.noise0_profiles == {}
    runtime.cleanup()


def test_profile_release_keeps_only_failures_and_retries_them():
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {}, platform="rk3576")
    class Context:
        def __init__(self, name, fail_once): self.name=name; self.fail_once=fail_once; self.calls=0
        def release(self):
            self.calls += 1
            if self.fail_once and self.calls == 1: raise RuntimeError(f"{self.name} transient")
    first, second, third = Context("T320", True), Context("T480", False), Context("T640", True)
    runtime.noise0_profiles = {320: first, 480: second, 640: third}
    with pytest.raises(RuntimeError, match="T640.*T320"):
        runtime.release_profile_contexts()
    assert runtime.noise0_profiles == {320: first, 640: third}
    assert second.calls == 1 and first.calls == third.calls == 1
    runtime.release_profile_contexts(); runtime.release_profile_contexts()
    assert runtime.noise0_profiles == {}
    assert second.calls == 1 and first.calls == third.calls == 2


def test_release_profile_contexts_waits_for_active_runtime_work():
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    entered = threading.Event(); unblock = threading.Event(); released = threading.Event()

    def active_request():
        with runtime._lifecycle_lock:
            entered.set()
            assert unblock.wait(2)

    worker = threading.Thread(target=active_request); worker.start()
    assert entered.wait(2)
    cleaner = threading.Thread(target=lambda: (runtime.release_profile_contexts(), released.set()))
    cleaner.start()
    assert not released.wait(.05)
    unblock.set(); worker.join(2); cleaner.join(2)
    assert released.is_set()


def test_runtime_info_reports_logical_and_loaded_contexts(tmp_path):
    _reset_mask_lite(); root = _complete_bundle(tmp_path)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, context_core_masks={"main0": 1, "noise1-convs1_0": 4})
    info = runtime.runtime_info()
    assert info["logical_contexts"] == info["contexts"] == 32
    assert info["core_mask_symbols"] == {"AUTO": 0, "CORE0": 1, "CORE1": 2, "CORE2": 4}
    assert info["context_core_masks"]["main0"] == 1 and len(info["context_core_masks"]) == 32
    assert info["active_core_masks"] == {"main0": 1, "noise1-convs1_0": 4}
    assert info["loaded_physical_contexts"] == 0
    runtime.preload(); assert runtime.runtime_info()["loaded_physical_contexts"] == 32
    runtime.cleanup()


def test_runtime_uses_caller_segments_without_resplitting(monkeypatch):
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    seen = []
    def synth_one(text, **_):
        seen.append(text)
        return np.ones(24, np.float32), {"duration_s": 0.001, "wall_ms": 1.0}
    monkeypatch.setattr(runtime, "_synthesize_one", synth_one)
    audio, meta = runtime.synthesize("ignored.ignored", segments=["raw?", "second.without split"])
    assert seen == ["raw?", "second.without split"]
    assert meta["segments"] == 2 and audio.size == 48


def test_runtime_cancel_stops_following_group_and_reports_partial(monkeypatch):
    runtime = KokoroLong32Runtime("/unused", frontend=lambda **_: {})
    cancel = threading.Event()
    seen = []
    def synth_one(text, **_):
        seen.append(text)
        cancel.set()
        return np.ones(24, np.float32), {"duration_s": 0.001, "wall_ms": 1.0}
    monkeypatch.setattr(runtime, "_synthesize_one", synth_one)
    audio, meta = runtime.synthesize("ignored", segments=["first", "must-not-run"], cancel_event=cancel)
    assert seen == ["first"]
    assert audio.size == 24 and meta["cancelled"] is True


def test_run_profile_cancel_drains_running_futures_before_next_request():
    """Cancellation after main0 must not let a later request overlap workers."""
    runtime = object.__new__(KokoroLong32Runtime)
    runtime.prefix_only = False; runtime.ready = True; runtime.platform = "rk3588"; runtime.route_ts = (400, 640)
    runtime.pool = ThreadPoolExecutor(max_workers=4)
    release_noise = threading.Event(); noise_started = threading.Event(); noise_count = {"n": 0}; noise_count_lock = threading.Lock()
    allow_main0_return = threading.Event()
    first_cancel = threading.Event(); first_returned = threading.Event()

    class Ctx:
        def __init__(self, name): self.name = name
        def run(self, *values):
            if self.name in {"noise0", "noise1"}:
                with noise_count_lock:
                    noise_count["n"] += 1
                    # At least one noise future must be running before the
                    # cancellation boundary.  The second initial future may
                    # remain queued until the first one is released; the
                    # future drain below must consume both states.
                    if noise_count["n"] >= 1: noise_started.set()
                assert release_noise.wait(2)
            if self.name == "main0":
                # Keep main0's future in flight long enough for both initial
                # noise futures to start.  Without this gate, the executor
                # may run main0 and the caller can observe cancellation before
                # the other queued futures have been scheduled, making the
                # drain assertion timing-dependent.
                first_cancel.set()
                assert allow_main0_return.wait(2)
            if self.name in {"main0", "noise0", "rb0", "rb1", "rb2"}:
                return np.zeros((1, 256, 4000), np.float32)
            if self.name == "noise1": return np.zeros((1, 128, 400), np.float32)
            if self.name == "main1": return np.zeros((1, 128, 400), np.float32)
            return np.zeros((1, 22, 24001), np.float32)

    class Branch:
        def run(self, *_): return np.zeros((1, 128, 320), np.float32), {}

    runtime.contexts = {name: Ctx(name) for name in ("main0", "noise0", "rb0", "rb1", "rb2", "main1", "post")}
    runtime.branches = {3: Branch(), 4: Branch(), 5: Branch()}
    runtime.noise = Ctx("noise1")
    runtime._noise0_for = lambda _t: runtime.contexts["noise0"]
    tensors = {"x": np.zeros((1, 512, 400), np.float32), "s": np.zeros((1, 128), np.float32), "har": np.zeros((1, 22, 24001), np.float32)}

    result = {}
    def first_request():
        try: runtime.run_profile(tensors, cancel_event=first_cancel)
        except Exception as exc: result["exc"] = exc
        finally: first_returned.set()

    worker = threading.Thread(target=first_request); worker.start()
    assert noise_started.wait(1) and first_cancel.wait(1)
    assert not first_returned.wait(0.1), "run_profile returned before running futures drained"
    release_noise.set()
    allow_main0_return.set()
    worker.join(2)
    assert not worker.is_alive() and isinstance(result.get("exc"), Long32Cancelled)

    # The next request is only allowed after the prior request has consumed all
    # worker results; it must complete normally with no stale blocked context.
    runtime.noise = Ctx("noise1")
    release_noise.set()
    conv, meta = runtime.run_profile(tensors)
    runtime.pool.shutdown(wait=True)
    assert conv.shape == (1, 22, 24001) and meta["T"] == 400
