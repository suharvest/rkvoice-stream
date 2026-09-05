import numpy as np
import pytest
import importlib.util
from pathlib import Path
import json

from rkvoice_stream.backends.tts.kokoro_long32 import KokoroLong32Runtime, _Branch
from rkvoice_stream.backends.tts.kokoro_convonly_prefix import OnlinePrefixAdapter
_spec = importlib.util.spec_from_file_location("long32_test_helpers", Path(__file__).with_name("test_kokoro_long32_runtime.py"))
_helpers = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_helpers)
_MaskLite, _complete_bundle, _reset_mask_lite = _helpers._MaskLite, _helpers._complete_bundle, _helpers._reset_mask_lite


@pytest.mark.parametrize("platform,expected", [("rk3588", 13), ("rk3576", 12)])
def test_prefix_only_loads_only_prefix_and_noise_contexts(tmp_path, platform, expected):
    _reset_mask_lite(); root = _complete_bundle(tmp_path, target=platform)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite,
                                  platform=platform, prefix_only=True)
    runtime.preload()
    assert len(_MaskLite.created) == expected
    names = {Path(x.path).name for x in _MaskLite.created}
    required = {f"{n}.fp16.rknn" for n in ("main0", "rb0", "rb1", "rb2", "main1", "noise1-preconv")}
    required |= {f"noise1-convs{h}_{i}.fp16.rknn" for h in (1, 2) for i in range(3)}
    if platform == "rk3588": required.add("noise0.fp16.rknn")
    assert names == required
    assert runtime.runtime_info()["prefix_only"] is True
    assert runtime.runtime_info()["loaded_physical_contexts"] == expected
    with pytest.raises(RuntimeError, match="prefix-only"):
        runtime.run_profile({})
    runtime.cleanup(); assert len(_MaskLite.released) == expected


def test_prefix_only_rk3576_lazy_noise_profiles_are_reused_and_cleanup_is_idempotent(tmp_path):
    _reset_mask_lite(); root = _complete_bundle(tmp_path, target="rk3576")
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, platform="rk3576", prefix_only=True)
    runtime.preload(); assert len(_MaskLite.created) == 12
    a = runtime._noise0_for(320); assert runtime._noise0_for(320) is a
    runtime._noise0_for(368); assert len(_MaskLite.created) == 14
    runtime.cleanup(); runtime.cleanup(); assert len(_MaskLite.released) == 14
    assert runtime.runtime_info()["loaded_physical_contexts"] == 0
    runtime.preload()
    assert runtime.runtime_info()["loaded_physical_contexts"] == 12
    runtime.cleanup()
    assert len(_MaskLite.created) == len(_MaskLite.released) == 26


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_prefix_only_requires_strict_bool(tmp_path, value):
    with pytest.raises(TypeError, match="prefix_only"):
        KokoroLong32Runtime(tmp_path, prefix_only=value)


def test_params_only_branch_rejects_run_before_math(tmp_path):
    root = _complete_bundle(tmp_path)
    branch = _Branch(_MaskLite, root, 3, load_contexts=False)
    with pytest.raises(RuntimeError, match="params-only"):
        branch.run(np.zeros((1, 256, 8), np.float32), np.zeros((1, 128), np.float32))
    branch.release()


def test_prefix_only_synthesize_rejected_before_empty_or_frontend(tmp_path):
    runtime = KokoroLong32Runtime(tmp_path, frontend=lambda **_: (_ for _ in ()).throw(AssertionError("frontend called")), prefix_only=True)
    with pytest.raises(RuntimeError, match="prefix-only"):
        runtime.synthesize("")


@pytest.mark.parametrize("platform,count", [("rk3588", 32), ("rk3576", 31)])
def test_legacy_default_load_inventory_unchanged(tmp_path, platform, count):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path, target=platform)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, platform=platform)
    try:
        runtime.preload()
        names = {Path(x.path).name for x in _MaskLite.created}
        expected = {f"{s}.fp16.rknn" for s in _helpers.CONTEXT_STAGES}
        if platform == "rk3576": expected.remove("noise0.fp16.rknn")
        assert names == expected and len(names) == count
        assert runtime.prefix_only is False
    finally: runtime.cleanup()


def test_nonzero_film_parameters_preserve_site_order(tmp_path):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path)
    # Distinct nonzero values for every block/half/stage and output channel.
    for block in (3, 4, 5):
        for half in (1, 2):
            for stage in range(3):
                stem = root / "params" / "rb345" / f"rb{block}-adain{half}_{stage}"
                key = block * 100 + half * 10 + stage
                weight = np.full((256, 128), key / 1024, np.float32)
                bias = np.arange(256, dtype=np.float32) / 512 + key
                weight.tofile(str(stem) + "-fc-weight.bin")
                bias.tofile(str(stem) + "-fc-bias.bin")
    full = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite)
    prefix = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, prefix_only=True)
    try:
        full.preload(); prefix.preload()
        style = np.arange(128, dtype=np.float32).reshape(1, 128) / 128
        actual = OnlinePrefixAdapter(prefix)._film(style)
        reference = OnlinePrefixAdapter(full)._film(style)
        for a, b in zip(actual, reference): np.testing.assert_array_equal(a, b)
        expected_g, expected_b = [], []
        for block in (3, 4, 5):
            assert not prefix.branches[block].ctx
            assert len(prefix.branches[block].params) == 6
            for stage in range(3):
                for half in (1, 2):
                    key = block * 100 + half * 10 + stage
                    w = np.full((256, 128), key / 1024, np.float32)
                    b = np.arange(256, dtype=np.float32) / 512 + key
                    values = style @ w.T + b
                    expected_g.append(values[:, :128]); expected_b.append(values[:, 128:])
        np.testing.assert_array_equal(actual[0], np.stack(expected_g, axis=1))
        np.testing.assert_array_equal(actual[1], np.stack(expected_b, axis=1))
    finally:
        prefix.cleanup(); full.cleanup()
    assert len(_MaskLite.created) == len(_MaskLite.released)


@pytest.mark.parametrize("name", ["post.fp16.rknn", "rb5-convs2_2.fp16.rknn"])
def test_unloaded_models_still_hash_checked(tmp_path, name):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path)
    (root / "models" / name).write_bytes(b"tampered")
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, prefix_only=True)
    with pytest.raises(ValueError, match="SHA"):
        runtime.preload()
    assert not _MaskLite.created


def test_prefix_requires_complete_rk3576_smoke(tmp_path):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path, target="rk3576")
    path = root / "device-smoke-report.json"
    obj = json.loads(path.read_text()); obj["passed_runs"] = 191
    path.write_text(json.dumps(obj))
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=_MaskLite, platform="rk3576", prefix_only=True)
    with pytest.raises(ValueError, match="coverage"):
        runtime.preload()
    assert not _MaskLite.created


@pytest.mark.parametrize("fault", ["used_context", "params", "noise", "executor"])
def test_prefix_partial_failures_release_every_created_context(tmp_path, monkeypatch, fault):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path)
    class Failing(_MaskLite):
        def init_runtime(self, **kwargs):
            failed = {"used_context": "rb1.fp16.rknn", "noise": "noise1-convs2_0.fp16.rknn"}.get(fault)
            if Path(self.path).name == failed: raise RuntimeError("injected init failure")
            return super().init_runtime(**kwargs)
    if fault == "params":
        (root / "params" / "rb345" / "rb4-adain1_0-fc-bias.bin").write_bytes(b"invalid")
    if fault == "executor":
        def fail_pool(**kwargs): raise RuntimeError("injected executor failure")
        monkeypatch.setattr("rkvoice_stream.backends.tts.kokoro_long32.ThreadPoolExecutor", fail_pool)
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=Failing, prefix_only=True)
    with pytest.raises((ValueError, RuntimeError)):
        runtime.preload()
    assert len(Failing.created) > 0
    assert len(Failing.created) == len(Failing.released)
    assert len({id(c) for c in Failing.released}) == len(Failing.released)
    assert not runtime.ready and not runtime.contexts and not runtime.branches and runtime.noise is None
    runtime.cleanup()
    assert len(Failing.created) == len(Failing.released)


@pytest.mark.parametrize("platform", ["rk3588", "rk3576"])
def test_run_prefix_nonzero_boundary_parity(tmp_path, platform):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path, target=platform)
    class Operators(_MaskLite):
        def inference(self, inputs):
            name = Path(self.path).name
            x = inputs[0]
            if name.startswith("noise0"):
                # Same deterministic expansion for dynamic/static noise0.
                f = inputs[1].shape[-1]
                out = np.full((1, 256, (f-1)//6), x.mean()+0.2, np.float32)
            elif name.startswith("main0"):
                out = np.repeat(x[:, :256, :], 10, axis=2) * np.float32(0.75)
            elif name.startswith("main1"):
                out = np.repeat(x[:, :128, :], 6, axis=2)
                out = np.concatenate([out, out[:, :, -1:]], axis=2)
            elif name.startswith("noise1-preconv"):
                out = np.repeat(x.mean(axis=1, keepdims=True), 128, axis=1)
            elif name.startswith("noise1-convs"):
                out = x * np.float32(0.5) + np.float32(0.1)
            elif name.startswith(("rb0", "rb1", "rb2")):
                out = x + inputs[1].mean() + np.float32(int(name[2])/10)
            else:
                raise AssertionError(f"unexpected inference: {name}")
            return [np.ascontiguousarray(out, np.float32)]
    full = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=Operators, platform=platform)
    prefix = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=Operators, platform=platform, prefix_only=True)
    try:
        full.preload(); prefix.preload()
        # Exercise the actual same params-only/full parameter lists with nonzero FiLM.
        for runtime in (full, prefix):
            for block, branch in runtime.branches.items():
                for site, (weight, bias, alpha) in enumerate(branch.params):
                    weight.fill((block+site+1)/1000); bias[:] = np.arange(256)/1000
        t = 320
        tensors = {"x": np.linspace(-0.3, 0.5, 512*t, dtype=np.float32).reshape(1,512,t),
                   "s": np.linspace(-0.1, 0.4, 128, dtype=np.float32).reshape(1,128),
                   "har": np.linspace(-0.2, 0.3, 22*(60*t+1), dtype=np.float32).reshape(1,22,60*t+1)}
        a = OnlinePrefixAdapter(full).run_prefix(tensors)
        b = OnlinePrefixAdapter(prefix).run_prefix(tensors)
        for name in ("j1", "gamma", "beta"):
            np.testing.assert_array_equal(getattr(a,name), getattr(b,name))
            assert np.any(getattr(b,name)) and np.isfinite(getattr(b,name)).all()
        assert b.j1.shape == (1,128,60*t+1)
    finally:
        prefix.cleanup(); full.cleanup()
    assert len(Operators.created) == len(Operators.released)


@pytest.mark.parametrize("failed_name", ["rb1.fp16.rknn", "noise1-convs2_0.fp16.rknn", "noise0.t320.fp16.rknn"])
def test_prefix_cleanup_failure_drains_all_owned_contexts(tmp_path, failed_name):
    _reset_mask_lite()
    root = _complete_bundle(tmp_path, target="rk3576")
    class ReleaseFailure(_MaskLite):
        attempts = {}
        def release(self):
            # Static noise0 filenames encode profile differently between bundles.
            name = Path(self.path).name
            self.attempts[name] = self.attempts.get(name, 0) + 1
            matches = name == failed_name or (failed_name.startswith("noise0.") and "noise0" in name)
            if matches and self.attempts[name] == 1:
                raise RuntimeError("injected release failure")
            super().release()
    runtime = KokoroLong32Runtime(root, frontend=lambda **_: {}, lite_cls=ReleaseFailure, platform="rk3576", prefix_only=True)
    runtime.preload(); runtime._noise0_for(320)
    with pytest.raises(RuntimeError, match="long32 cleanup errors.*injected release failure"):
        runtime.cleanup()
    assert len(ReleaseFailure.created) == 13 and len(ReleaseFailure.released) == 12
    assert not runtime.ready and runtime.pool is None
    assert bool(runtime.contexts) + bool(runtime.noise is not None) + bool(runtime.noise0_profiles) == 1
    runtime.cleanup()
    assert len(ReleaseFailure.released) == 13
    assert len({id(c) for c in ReleaseFailure.released}) == 13
    assert not runtime.contexts and not runtime.branches
    assert runtime.noise is None and not runtime.noise0_profiles
    runtime.cleanup()
    assert len(ReleaseFailure.released) == 13
