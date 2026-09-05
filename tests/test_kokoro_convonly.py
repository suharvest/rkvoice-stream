"""Host-only orchestration tests: real bundle gates/router/iSTFT/WAV, fake models.

Synthetic model bytes below are not deployable or qualification receipts.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import wave

from rkvoice_stream.backends.tts import kokoro_convonly as M
from rkvoice_stream.backends.tts.kokoro_long32 import KokoroLong32Runtime
from rkvoice_stream.backends.tts.kokoro_long32_frontend import KokoroLong32Frontend
from rkvoice_stream.engine.tts import TTSBackend

FULL = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
GENERATOR = "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c"


def test_source_planner_uses_phoneme_measure_and_preserves_email_decimal(tmp_path, models):
    text = "Contact x@y.com or use 3.14 value."
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    def prepare(part, **kw):
        if len(part) > 18: raise M.PhonemeLengthError("long")
        return SimpleNamespace(natural_D=160, seed=len(part))
    backend.frontend.prepare = prepare
    got = list(backend._iter_planned_segments(text, "en", 1, ()))
    assert "".join(text[a:b] for a, b, _, _ in got) == text
    assert any("x@y.com" in text[a:b] for a, b, _, _ in got)
    assert any("3.14" in text[a:b] for a, b, _, _ in got)
    backend.close()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _pin(root, manifest=None, *, inventory=True):
    path = root / M.MANIFEST_NAME
    manifest = json.loads(path.read_text()) if manifest is None else manifest
    if inventory:
        manifest["files"] = [{"path": p.relative_to(root).as_posix(), "sha256": M._sha(p), "size": p.stat().st_size}
                             for p in sorted(root.rglob("*")) if p.is_file() and p != path]
    _write(path, manifest)
    return M.KokoroConvOnlyConfig(manifest["platform"], root, M._sha(path))


def _bundle(root, platform="rk3588"):
    front, prefix = root / "frontend", root / "prefix"
    front.mkdir(parents=True)
    (prefix / "models").mkdir(parents=True)
    export = {"schema": "kokoro-v10-generator-export-1", "status": "PASS", "target": "rk3588",
              "full_model_sha256": FULL, "output_sha256": GENERATOR}
    for directory in (front, prefix):
        _write(directory / "generator-export.manifest.json", export)
    export_sha = M._sha(front / "generator-export.manifest.json")
    _write(front / "tokens.json", {"h": 1})
    for voice in M.VOICES.values():
        (front / f"{voice}.bin").write_bytes(b"host-only-voice")
    for name in ("frontend_stem.onnx", "duration_head.onnx", "target_tail.onnx"):
        (front / name).write_bytes(b"host-only-model")
    fm = {"schema": "kokoro-v10-long32-frontend-3", "status": "PASS", "official": True,
          "routes": M.ROUTES, "voices": M.VOICES, "full_source_sha256": FULL,
          "generator_source_sha256": GENERATOR, "generator_export_manifest_sha256": export_sha,
          "generator_export_manifest": {"path": "generator-export.manifest.json", "schema": export["schema"], "status": "PASS", "sha256": export_sha},
          "stem_model": "frontend_stem.onnx", "duration_model": "duration_head.onnx", "target_model": "target_tail.onnx",
          "stem_sha256": M._sha(front / "frontend_stem.onnx"), "duration_sha256": M._sha(front / "duration_head.onnx"),
          "target_sha256": M._sha(front / "target_tail.onnx"),
          "assets": {p.name: {"bytes": p.stat().st_size, "sha256": M._sha(p)} for p in front.iterdir() if p.suffix != ".onnx"}}
    _write(front / "frontend.manifest.json", fm)
    rows = []
    ts = M.PLATFORM_TS[platform]
    for stage in M.CONTEXT_STAGES:
        row = {"stage": stage, "status": "PASS"}
        if platform == "rk3576" and stage == "noise0":
            static = {}
            for t in ts:
                name = f"noise0-T{t}.fp16.rknn"
                (prefix / "models" / name).write_bytes(name.encode())
                static[f"T{t}"] = {"T": t, "F": 60*t+1, "L": 10*t, "rknn": name,
                    "rknn_sha256": M._sha(prefix / "models" / name), "inputs": [[1,128],[1,22,60*t+1]],
                    "outputs": [[1,256,10*t]], "input_count": 2, "output_count": 1}
            row["static_profiles"] = static
        else:
            name = f"{stage}.fp16.rknn"
            (prefix / "models" / name).write_bytes(name.encode())
            row["rknn_sha256"] = M._sha(prefix / "models" / name)
        rows.append(row)
    parameters = []
    for group, blocks in (("rb345", ("rb3", "rb4", "rb5")), ("noise1", ("noise1",))):
        for block in blocks:
            for half in (1, 2):
                for index in range(3):
                    for suffix, count in (("fc-weight", 32768), ("fc-bias", 256), ("alpha", 128)):
                        path = prefix / "params" / group / f"{block}-adain{half}_{index}-{suffix}.bin"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.ones(count, dtype=np.float32).tofile(path)
                        parameters.append({"group": group, "path": f"/original-source/{path.name}",
                                           "sha256": M._sha(path), "bytes": path.stat().st_size})
    pm = {"schema": f"kokoro-v10-{platform}-long-hybrid-full-v7", "status": "PASS", "contexts": 32,
          "target": platform, "precision": "FP16", "toolkit": "2.3.2", "rows": rows,
          "profiles": {f"T{t}": {"T": t, "L": 10*t, "F": 60*t+1} for t in ts},
          "full_source_sha256": FULL, "generator_source_sha256": GENERATOR,
          "source_sha256": GENERATOR, "source_sha256_deprecated": True, "source_sha256_kind": "generator_model",
          "generator_export_manifest_sha256": export_sha,
          "preflight_evidence": {"generator_export_manifest_sha256": export_sha}, "lineage": {"parameters": parameters}}
    _write(prefix / "build-manifest.json", pm)
    if platform == "rk3576":
        smoke = {"schema": "kokoro-v10-isolated-smoke-2", "status": "PASS", "target": platform,
                 "full_source_sha256": FULL, "generator_source_sha256": GENERATOR,
                 "generator_export_manifest_sha256": export_sha,
                 "build_manifest_sha256": M._sha(prefix / "build-manifest.json"),
                 "required_runs": 192, "passed_runs": 192, "profiles": list(ts), "models": []}
        for row in rows:
            stage = row["stage"]
            runs = []
            for t in ts:
                shape = [1, 22, 60*t+1] if stage == "post" else ([1, 128, 60*t+1] if stage == "main1" or stage.startswith(("rb3-", "rb4-", "rb5-", "noise1-")) else [1, 256, 10*t])
                runs.append({"T": t, "rknn_sha256": row["static_profiles"][f"T{t}"]["rknn_sha256"] if stage == "noise0" else row["rknn_sha256"],
                             "output_shapes": [shape], "load_return": 0, "init_return": 0,
                             "finite": True, "release_success": True, "inference_s": .01})
            smoke["models"].append({"stage": stage, "runs": runs})
        _write(prefix / "device-smoke-report.json", smoke)
    (root / "tail" / "models").mkdir(parents=True)
    (root / "tail" / "slopes").mkdir()
    for b in range(3):
        for u in range(3):
            for c in (1, 2):
                stem = f"branch{b}_unit{u}_conv{c}"
                (root / "tail" / "models" / f"{stem}.convonly.{platform}.fp16.rknn").write_bytes(b"conv")
                np.save(root / "tail" / "slopes" / f"{stem}.npy", np.ones(128, np.float32))
    merge = f"tail/merge-official-v9.{platform}.fp16.rknn"
    (root / merge).write_bytes(b"merge")
    (root / "tail" / "library.so").write_bytes(b"host-only-never-loaded")
    return _pin(root, {"schema": "kokoro.convonly.bundle.v1", "platform": platform, "precision": "FP16",
        "full_source_sha256": FULL, "generator_source_sha256": GENERATOR, "frontend": "frontend", "prefix": "prefix",
        "tail": {"model_dir": "tail/models", "slopes_dir": "tail/slopes", "merge": merge,
                 "library": "tail/library.so", "implementation": "native" if platform == "rk3576" else "python"}})


@pytest.fixture
def models(monkeypatch):
    state = SimpleNamespace(calls=[], natural=160, fail=None, cleanup_fail=set(), cancel=None,
                            block_enter=None, block_release=None, bad_boundary=False, bad_tail=False)
    def call(name):
        state.calls.append(name)
        if state.cancel and state.cancel[0] == name:
            state.cancel[1].set()
        if state.fail == name:
            raise RuntimeError(f"injected {name}")
    class Frontend(KokoroLong32Frontend):
        @classmethod
        def _session_options_factory(cls, intra, inter):
            state.threads = (intra, inter)
            return None
        def __init__(self, *a, **kw):
            call("frontend.create")
            super().__init__(*a, **kw)  # Real schema/provenance/vocabulary gates.
        def preload_sessions(self):
            call("frontend.preload")
        def prepare(self, text, *, language, speed, cancel_check=None):
            call("prepare")
            return SimpleNamespace(natural_D=state.natural, seed=len(text))
        def render(self, prepared, *, target_D):
            call("render")
            return {"N": 120*target_D+1, "seed": prepared.seed, "frontend_detail": {"fake_model": 1}}
    class Runtime(KokoroLong32Runtime):
        def __init__(self, *a, **kw):
            call("runtime.create")
            state.runtime_kwargs = kw
            super().__init__(*a, **kw)
        def _manifest(self):
            call("runtime.manifest")
            return super()._manifest()
        def preload(self):
            call("runtime.preload")
            self.ready = True
        def cleanup(self):
            call("runtime.cleanup")
            self.ready = False
            if "runtime" in state.cleanup_fail:
                raise RuntimeError("runtime cleanup failed")
    class Prefix:
        def __init__(self, runtime):
            call("prefix.create")
            self.runtime = runtime
        def run_prefix(self, rendered):
            call("prefix")
            shape = (1, 128, rendered["N"] + int(state.bad_boundary))
            return SimpleNamespace(j1=np.full(shape, rendered["seed"], np.float32),
                gamma=np.zeros((1,18,128), np.float32), beta=np.zeros((1,18,128), np.float32), timings={"fake_model": 1})
    class Tail:
        def __init__(self, **kw):
            call("tail.create")
            state.tail_kwargs = kw
        def run(self, j1, gamma, beta):
            call("tail")
            if state.block_enter:
                state.block_enter.set()
                assert state.block_release.wait(3)
            state.tail_lengths = getattr(state, "tail_lengths", []) + [j1.shape[-1]]
            out = np.zeros((1,22,j1.shape[-1]), np.float32)
            out[:, 0, :] = j1[0, 0, 0] / 100
            if state.bad_tail:
                out[0, 0, 0] = np.nan
            return out, {"model_test_only": True}
        def close(self):
            call("tail.close")
            if "tail" in state.cleanup_fail:
                raise RuntimeError("tail cleanup failed")
    monkeypatch.setattr(M, "KokoroLong32Frontend", Frontend)
    monkeypatch.setattr(M, "KokoroLong32Runtime", Runtime)
    monkeypatch.setattr(M, "OnlinePrefixAdapter", Prefix)
    monkeypatch.setattr(M, "PythonConvOnlyTail", Tail)
    monkeypatch.setattr(M, "NativeConvOnlyTail", Tail)
    return state


@pytest.mark.parametrize("platform,intra", [("rk3588", 4), ("rk3576", 6)])
def test_full_real_orchestration_and_variable_length_wav(tmp_path, models, platform, intra, monkeypatch):
    cfg = _bundle(tmp_path, platform)
    monkeypatch.setenv("KOKORO_LONG32_ROUTE_TS", "bad,legacy,env")
    monkeypatch.setenv("KOKORO_LONG32_MAX_SNAP_RATIO", "1")
    backend = M.KokoroConvOnlyBackend(cfg)
    assert isinstance(backend, TTSBackend)
    assert not backend.is_ready()
    data, meta = backend.synthesize("hello", language="en-US", voice="af_heart")
    assert backend.is_ready()
    assert models.calls == ["frontend.create", "runtime.create", "runtime.manifest", "frontend.preload", "runtime.preload", "prefix.create", "tail.create", "prepare", "render", "prefix", "tail"]
    assert models.threads == (intra, 1)
    assert models.runtime_kwargs["route_ts"] == M.PLATFORM_TS[platform]
    assert models.runtime_kwargs["context_core_masks"] == {}
    assert models.runtime_kwargs["prefix_only"] is True
    assert models.tail_kwargs["slopes"].shape == (18,128)
    assert models.tail_kwargs["max_n"] == 38401
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 24000, 96000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
    q = np.zeros((1,22,19201), np.float32)
    q[:, 0, :] = np.float32(.05)
    expected = np.clip(np.rint(M.conv_post_to_waveform(q)[0]*32767), -32768, 32767).astype("<i2")
    np.testing.assert_array_equal(pcm, expected)
    assert meta["audio_s"] == 4
    assert meta["full_rtf"] == pytest.approx(meta["total_ms"] / 4000)
    assert meta["tail_rtf"] == pytest.approx(meta["tail_ms"] / 4000)
    assert meta["total_ms"] >= sum(meta[k] for k in ("frontend_ms", "prefix_ms", "tail_ms", "istft_ms", "wav_ms"))
    assert meta["manifest_sha256"] == cfg.manifest_sha256
    assert not meta["fallback"] and not meta["cancelled"]
    models.natural = 240
    other, _ = backend.synthesize("different text")
    assert other != data
    assert models.tail_lengths == [19201, 28801]
    assert models.calls.count("tail.create") == 1
    runtime = backend.prefix_runtime
    backend.close(); backend.close()
    assert runtime.frontend is None
    assert backend.frontend is backend.prefix is backend.tail is backend.prefix_runtime is None
    assert models.calls.count("tail.close") == models.calls.count("runtime.cleanup") == 1
    assert not backend.is_ready()


def test_is_ready_is_nonblocking_advisory_snapshot():
    backend = M.KokoroConvOnlyBackend(M.KokoroConvOnlyConfig(
        "rk3588", Path("/tmp/kokoro-convonly-test-bundle"),
        "0" * 64,
    ))
    entered = threading.Event()
    release = threading.Event()

    def hold_lifecycle_lock():
        with backend._lock:
            entered.set()
            release.wait(timeout=2)

    holder = threading.Thread(target=hold_lifecycle_lock)
    holder.start()
    try:
        assert entered.wait(timeout=1)
        started = __import__("time").monotonic()
        assert not backend.is_ready()
        assert __import__("time").monotonic() - started < 0.1
    finally:
        release.set()
        holder.join(timeout=2)
    assert not holder.is_alive()

    backend._ready = True
    assert backend.is_ready()
    backend._closed = True
    assert not backend.is_ready()


def test_single_chunk_stream_uses_complete_wav(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    chunks = list(backend.synthesize_stream("hello"))
    assert backend.supports_streaming and len(chunks) == 100
    assert all(chunk[0].dtype == np.dtype("float32") for chunk in chunks)
    assert all(np.max(np.abs(chunk[0])) <= 1.0 for chunk in chunks)
    assert sum(len(chunk[0]) for chunk in chunks) == 96000
    backend.close()


@pytest.mark.parametrize("kw", [{"intra": True}, {"intra": -1}, {"intra": 1.2}, {"intra": "4"}, {"inter": 0}, {"inter": False}, {"inter": 2**31}, {"manifest_sha256": "g"*64}, {"manifest_sha256": "a"*63}, {"bundle_root": "relative"}, {"platform": "bad"}])
def test_strict_explicit_config(tmp_path, kw):
    args = dict(platform="rk3588", bundle_root=tmp_path, manifest_sha256="a"*64)
    args.update(kw)
    with pytest.raises((ValueError, TypeError)):
        M.KokoroConvOnlyConfig(**args)


@pytest.mark.parametrize("raw", ["", "-1", "1.0", " 4", "True", "nan"])
def test_thread_env_is_strict(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("KOKORO_CONVONLY_ROOT", str(tmp_path))
    monkeypatch.setenv("KOKORO_CONVONLY_MANIFEST_SHA256", "a"*64)
    monkeypatch.setenv("KOKORO_FRONTEND_INTRA_OP_THREADS", raw)
    with pytest.raises(ValueError):
        M.KokoroConvOnlyConfig.from_env()


def test_explicit_config_ignores_env_and_env_constructor_reads_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("RK_PLATFORM", "rk3576")
    monkeypatch.setenv("KOKORO_CONVONLY_ROOT", str(tmp_path))
    monkeypatch.setenv("KOKORO_CONVONLY_MANIFEST_SHA256", "a"*64)
    monkeypatch.setenv("KOKORO_FRONTEND_INTRA_OP_THREADS", "0")
    monkeypatch.setenv("KOKORO_FRONTEND_INTER_OP_THREADS", "2")
    before = dict(os.environ)
    assert (M.KokoroConvOnlyBackend().config.intra, M.KokoroConvOnlyBackend().config.inter) == (0, 2)
    explicit = M.KokoroConvOnlyBackend(M.KokoroConvOnlyConfig("rk3588", tmp_path, "a"*64))
    assert (explicit.config.intra, explicit.config.inter) == (4, 1)
    assert dict(os.environ) == before


@pytest.mark.parametrize("field,value", [("platform", "rk3576"), ("schema", "bad"), ("precision", "INT8"), ("full_source_sha256", "f"*64), ("generator_source_sha256", "0"*64)])
def test_manifest_identity_fails_before_component_creation(tmp_path, models, field, value):
    _bundle(tmp_path)
    obj = json.loads((tmp_path / M.MANIFEST_NAME).read_text()); obj[field] = value
    cfg = _pin(tmp_path, obj)
    # Keep target pinned to the original so a changed manifest target is rejected.
    cfg = M.KokoroConvOnlyConfig("rk3588", tmp_path, cfg.manifest_sha256)
    with pytest.raises(ValueError, match="identity"):
        M.KokoroConvOnlyBackend(cfg).preload()
    assert not models.calls


@pytest.mark.parametrize("change", ["wrong_digest", "tamper", "omitted", "extra", "duplicate", "size_bool", "sha_nonhex", "traversal", "absolute", "symlink", "schema2", "tail_impl", "lineage", "slope"])
def test_invalid_bundle_fails_before_npu(tmp_path, models, change):
    cfg = _bundle(tmp_path)
    obj = json.loads((tmp_path / M.MANIFEST_NAME).read_text())
    if change == "wrong_digest":
        cfg = M.KokoroConvOnlyConfig("rk3588", tmp_path, "0"*64)
    elif change == "tamper":
        (tmp_path / "tail/library.so").write_bytes(b"modified")
    elif change == "extra":
        (tmp_path / "unlisted-sidecar").write_bytes(b"extra")
    elif change == "symlink":
        (tmp_path / "unlisted-link").symlink_to(tmp_path / "tail/library.so")
    elif change in {"schema2", "lineage", "slope"}:
        if change == "schema2":
            path = tmp_path / "frontend/frontend.manifest.json"
            fm = json.loads(path.read_text()); fm["schema"] = "kokoro-v10-long32-frontend-2"; _write(path, fm)
        elif change == "lineage":
            path = tmp_path / "prefix/build-manifest.json"
            pm = json.loads(path.read_text()); pm["lineage"]["parameters"].pop(); _write(path, pm)
        else:
            np.save(tmp_path / "tail/slopes/branch0_unit0_conv1.npy", np.full(128, np.nan, np.float32))
        cfg = _pin(tmp_path)
    else:
        if change == "omitted": obj["files"].pop()
        if change == "duplicate": obj["files"].append(obj["files"][0])
        if change == "size_bool": obj["files"][0]["size"] = True
        if change == "sha_nonhex": obj["files"][0]["sha256"] = "z"*64
        if change == "traversal": obj["frontend"] = "tail/../frontend"
        if change == "absolute": obj["frontend"] = str(tmp_path / "frontend")
        if change == "tail_impl": obj["tail"]["implementation"] = "native"
        cfg = _pin(tmp_path, obj, inventory=False)
    with pytest.raises((ValueError, OSError)):
        M.KokoroConvOnlyBackend(cfg).preload()
    assert not any(x in models.calls for x in ("runtime.preload", "tail.create"))


def test_nested_component_gate_is_not_bypassed(tmp_path, models):
    _bundle(tmp_path)
    path = tmp_path / "prefix/build-manifest.json"
    pm = json.loads(path.read_text()); pm["status"] = "FAIL"; _write(path, pm)
    backend = M.KokoroConvOnlyBackend(_pin(tmp_path))
    with pytest.raises(ValueError, match="manifest"):
        backend.preload()
    assert "runtime.manifest" in models.calls and "runtime.preload" not in models.calls
    assert backend.frontend is None and not backend.is_ready()


@pytest.mark.parametrize("failure", ["frontend.preload", "runtime.preload", "prefix.create", "tail.create"])
def test_partial_initialization_unwinds_and_never_ready(tmp_path, models, failure):
    models.fail = failure
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    with pytest.raises(RuntimeError, match="injected"):
        backend.preload()
    assert not backend.is_ready() and backend.frontend is None
    assert models.calls.count("runtime.cleanup") == 1
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.preload()


def test_cleanup_errors_attempt_all_resources_and_remain_visible(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    models.cleanup_fail = {"tail", "runtime"}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="tail.*runtime"):
            backend.close()
    assert models.calls.count("tail.close") == models.calls.count("runtime.cleanup") == 1
    assert backend.frontend is None and not backend.is_ready()
    assert len(backend.runtime_info()["cleanup_errors"]) == 2


def test_preload_preserves_original_and_cleanup_errors(tmp_path, models):
    models.fail = "tail.create"; models.cleanup_fail = {"runtime"}
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    with pytest.raises(RuntimeError, match="injected tail.create.*cleanup errors.*runtime"):
        backend.preload()
    assert backend.frontend is None


@pytest.mark.parametrize("stage", ["prepare", "render", "prefix", "tail"])
def test_cancellation_stops_before_next_stage(tmp_path, models, stage):
    cancel = threading.Event(); models.cancel = (stage, cancel)
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    with pytest.raises(M.ConvOnlyCancelled) as exc:
        backend.synthesize("hello", cancel_event=cancel)
    assert exc.value.stage == stage
    stages = ["prepare", "render", "prefix", "tail"]
    assert [x for x in models.calls if x in stages] == stages[:stages.index(stage)+1]
    backend.close()


def test_precancel_does_not_load_models(tmp_path, models):
    cancel = threading.Event(); cancel.set()
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    with pytest.raises(M.ConvOnlyCancelled): backend.synthesize("hello", cancel_event=cancel)
    assert not models.calls


def test_close_waits_for_cancelled_native_work(tmp_path, models):
    cancel = threading.Event()
    models.block_enter = threading.Event(); models.block_release = threading.Event()
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    close_started = threading.Event()
    def close():
        close_started.set(); backend.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        run = pool.submit(backend.synthesize, "hello", cancel_event=cancel)
        assert models.block_enter.wait(2)
        cancel.set(); closing = pool.submit(close)
        assert close_started.wait(2)
        assert not closing.done() and "tail.close" not in models.calls
        models.block_release.set()
        with pytest.raises(M.ConvOnlyCancelled): run.result(timeout=3)
        closing.result(timeout=3)
    assert models.calls.index("tail") < models.calls.index("tail.close")


def test_concurrent_requests_serialize(tmp_path, models):
    models.block_enter = threading.Event(); models.block_release = threading.Event()
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    submitted = threading.Event()
    def second():
        submitted.set(); return backend.synthesize("second")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.synthesize, "first")
        assert models.block_enter.wait(2)
        other = pool.submit(second); assert submitted.wait(2)
        assert models.calls.count("prepare") == 1
        models.block_release.set(); first.result(timeout=3); other.result(timeout=3)
    assert models.calls.count("tail") == 2
    backend.close()


@pytest.mark.parametrize("kw", [{"speaker_id": 1}, {"pitch_shift": 1}, {"voice": "wrong"}, {"language": "xx"}, {"speed": True}, {"speed": float("nan")}, {"max_snap_ratio": 1}, {"cancel_event": object()}])
def test_unsupported_request_options_do_not_load(tmp_path, models, kw):
    backend = M.KokoroConvOnlyBackend(M.KokoroConvOnlyConfig("rk3588", tmp_path, "a"*64))
    with pytest.raises(ValueError): backend.synthesize("hello", **kw)
    assert not models.calls


def test_snap_cap_and_bad_boundary_cannot_reach_tail(tmp_path, models, monkeypatch):
    monkeypatch.setenv("KOKORO_LONG32_MAX_SNAP_RATIO", "1")
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); models.natural = 20
    with pytest.raises(ValueError, match="10%"):
        backend.synthesize("hello")
    assert "render" not in models.calls and "tail" not in models.calls
    models.natural = 160; models.bad_boundary = True
    with pytest.raises(ValueError, match="boundary"):
        backend.synthesize("hello")
    assert "tail" not in models.calls
    models.bad_boundary = False; models.bad_tail = True
    with pytest.raises(ValueError, match="tail output"):
        backend.synthesize("hello")
    backend.close()


def test_explicit_prefix_routes_ignore_legacy_env_and_none_preserves_it(tmp_path, monkeypatch):
    monkeypatch.setenv("KOKORO_LONG32_ROUTE_TS", "400,640")
    assert KokoroLong32Runtime(tmp_path, platform="rk3588").route_ts == (400,640)
    assert KokoroLong32Runtime(tmp_path, platform="rk3588", route_ts=(320,480)).route_ts == (320,480)
    with pytest.raises(ValueError):
        KokoroLong32Runtime(tmp_path, platform="rk3576", route_ts=(400,640))


def _facade(monkeypatch, backend):
    # Prefer the actual authorized sibling source checkout in this workspace.
    # CI without VoxEdge explicitly skips this optional cross-project test.
    source = Path(__file__).resolve().parents[4] / "voxedge"
    if (source / "voxedge/backends/rk/tts.py").is_file():
        monkeypatch.syspath_prepend(str(source))
    facade_module = pytest.importorskip("voxedge.backends.rk.tts", reason="VoxEdge source/runtime unavailable")
    if (source / "voxedge/backends/rk/tts.py").is_file():
        assert Path(facade_module.__file__).resolve() == source / "voxedge/backends/rk/tts.py"
    facade = facade_module.RKTTSBackend()
    facade._inner = backend
    return facade


@pytest.mark.parametrize("entry", ["synthesize", "synthesize_stream", "generate_streaming"])
def test_real_voxedge_facade_to_backend_wav_and_stream_contract(tmp_path, models, monkeypatch, entry):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    facade = _facade(monkeypatch, backend)
    assert facade.sample_rate == 24000 and facade.is_ready()
    if entry == "synthesize":
        data, metadata = facade.synthesize("hello", language="en", cancel_event=threading.Event())
        with wave.open(io.BytesIO(data)) as wav:
            assert wav.getframerate() == 24000 and wav.getnframes() == 96000
        assert metadata["backend"] == "kokoro_convonly"
    elif entry == "synthesize_stream":
        chunks = list(facade.synthesize_stream("hello", language="en", cancel_event=threading.Event()))
        assert len(chunks) == 100 and sum(len(chunk[0]) for chunk in chunks) == 96000
        assert chunks[0][1]["supports_streaming"] is True
    else:
        chunks = list(facade.generate_streaming("hello", language="en", speaker="af_heart", cancel_token=threading.Event()))
        assert len(chunks) == 100 and all(isinstance(chunk, bytes) and len(chunk) == 1920 for chunk in chunks)
    assert models.calls.count("tail") == 1
    facade.unload()
    assert not backend.is_ready() and models.calls.count("tail.close") == 1


@pytest.mark.parametrize("entry", ["synthesize", "synthesize_stream", "generate_streaming"])
def test_real_voxedge_cancellation_reaches_backend(tmp_path, models, monkeypatch, entry):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    facade = _facade(monkeypatch, backend)
    cancel = threading.Event(); models.cancel = ("prefix", cancel)
    key = "cancel_token" if entry == "generate_streaming" else "cancel_event"
    with pytest.raises(M.ConvOnlyCancelled, match="prefix"):
        result = getattr(facade, entry)("hello", **{key: cancel})
        if entry != "synthesize":
            list(result)
    assert "prefix" in models.calls and "tail" not in models.calls
    facade.unload()


def _install_stream_fake(backend, fake):
    """Exercise the real planner/stream path while replacing only model execution."""
    backend._lock = threading.RLock()
    backend.config = SimpleNamespace(platform="rk3588")
    backend.preload = lambda: None
    backend.frontend = SimpleNamespace(prepare=lambda text, **kw: SimpleNamespace(natural_D=160, text=text))
    def render(prepared, profile, route, voice, speed, cancel, start, end):
        payload, meta = fake(prepared.text)
        with wave.open(io.BytesIO(payload), "rb") as wav:
            audio_s = wav.getnframes() / wav.getframerate()
        return payload, {"frontend_ms": 0, "audio_s": audio_s, "source_start": start, "source_end": end, **meta}
    backend._render_segment = render


def test_sentence_stream_yields_pcm_before_next_sentence_and_preserves_text():
    backend = M.KokoroConvOnlyBackend.__new__(M.KokoroConvOnlyBackend)
    calls = []
    def fake(text, **kwargs):
        calls.append(text)
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x01\x00" * 1920)
        return b.getvalue(), {"backend":"kokoro_convonly", "total_ms":1}
    _install_stream_fake(backend, fake)
    stream = backend.synthesize_stream("First. Second!", language="en")
    first = next(stream)
    assert calls == ["First. "]
    chunks = [first, *list(stream)]
    assert calls == ["First. ", "Second!"]
    assert len(chunks) == 4 and all(chunk[0].dtype == np.dtype("float32") for chunk in chunks)
    assert chunks[0][1]["sentence_index"] == 0 and chunks[0][1]["chunk_index"] == 0
    assert all(chunk[1]["streaming_mode"] == "sentence" and not chunk[1]["model_incremental"] for chunk in chunks)
    assert not any(chunk[0].tobytes().startswith(b"RIFF") for chunk in chunks)


def test_sentence_stream_cancel_and_close_stop_following_synthesis():
    backend = M.KokoroConvOnlyBackend.__new__(M.KokoroConvOnlyBackend); calls=[]; cancel=threading.Event()
    def fake(text, **_):
        calls.append(text); b=io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x01\x00" * 960)
        return b.getvalue(), {"audio_s": 0.04}
    _install_stream_fake(backend, fake)
    stream = backend.synthesize_stream("First. Second.", language="en", cancel_event=cancel)
    next(stream)
    assert calls == ["First. "]
    cancel.set(); stream.close(); assert calls == ["First. "]


def test_sentence_stream_splits_cjk_without_space_and_propagates_second_error():
    backend = M.KokoroConvOnlyBackend.__new__(M.KokoroConvOnlyBackend); calls=[]
    def fake(text, **_):
        calls.append(text)
        if len(calls) == 2: raise RuntimeError("second sentence failed")
        b=io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x02\x00" * 960)
        return b.getvalue(), {"audio_s": 0.04}
    _install_stream_fake(backend, fake)
    stream = backend.synthesize_stream("第一句。第二句。", language="zh")
    first = next(stream)
    assert first[1]["sentence_text"] == "第一句。"
    with pytest.raises(RuntimeError, match="second sentence failed"):
        next(stream)
    assert calls == ["第一句。", "第二句。"]


def _pcm_stream_fake(frames=960, *, truncate=False):
    backend = M.KokoroConvOnlyBackend.__new__(M.KokoroConvOnlyBackend)
    calls = []
    pcm = np.resize(np.array([-32768, -1, 0, 1, 32767], dtype="<i2"), frames)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm.tobytes())
    payload = output.getvalue()
    if truncate:
        payload = payload[:-2]
    def synthesize(text, **kwargs):
        calls.append(text)
        return payload, {"audio_s": frames / 24000}
    _install_stream_fake(backend, synthesize)
    return backend, calls, pcm


def test_sentence_stream_t368_duration_and_pcm_are_bit_exact():
    backend, calls, pcm = _pcm_stream_fake(110400)
    chunks = list(backend.synthesize_stream("A valid test sentence."))
    assert len(chunks) == 115
    expected = pcm.astype(np.float32) / np.float32(32768.0)
    assert np.array_equal(np.concatenate([item[0] for item in chunks]), expected)
    assert chunks[-1][1]["segment_complete"] is True
    assert calls == ["A valid test sentence."]


def test_sentence_stream_rejects_truncated_wav_before_yield():
    backend, _, _ = _pcm_stream_fake(truncate=True)
    with pytest.raises(ValueError, match="frame count"):
        next(backend.synthesize_stream("A valid test sentence."))


@pytest.mark.parametrize("text", ["第一句。\n", "第一句。 \n\t"])
def test_sentence_stream_retains_cjk_trailing_whitespace_without_extra_call(text):
    backend, calls, _ = _pcm_stream_fake()
    list(backend.synthesize_stream(text, language="zh"))
    assert calls == [text]


@pytest.mark.parametrize("key", ["cancel_event", "cancel_token"])
def test_sentence_stream_cancel_on_resume_prevents_next_sentence(key):
    backend, calls, _ = _pcm_stream_fake()
    event = threading.Event()
    stream = backend.synthesize_stream("First. Second.", **{key: event})
    next(stream)
    event.set()
    with pytest.raises(M.ConvOnlyCancelled):
        next(stream)
    assert calls == ["First. "]


def test_sentence_stream_close_without_cancel_prevents_next_sentence():
    backend, calls, _ = _pcm_stream_fake()
    stream = backend.synthesize_stream("First. Second.")
    next(stream)
    stream.close()
    assert list(stream) == []
    assert calls == ["First. "]


def test_sentence_stream_preserves_internal_email_and_decimal_dots():
    backend, calls, _ = _pcm_stream_fake()
    text = "Email test@example.com costs 3.14 dollars. Another sentence."
    list(backend.synthesize_stream(text))
    assert calls == ["Email test@example.com costs 3.14 dollars. ", "Another sentence."]
    assert "".join(calls) == text


def _cpu_bundle(root, monkeypatch):
    cfg = _bundle(root)
    manifest = json.loads((root / M.MANIFEST_NAME).read_text())
    with (root / "cpu.onnx").open("wb") as fake:
        fake.write(b"host fake generator, never an ONNX model")
        fake.truncate(78971566)  # Sparse host-only fixture; digest is mocked below.
    manifest["cpu_generator"] = "cpu.onnx"
    real_sha = M._sha
    monkeypatch.setattr(M, "_sha", lambda p: GENERATOR if p == root / "cpu.onnx" else real_sha(p))
    return _pin(root, manifest)


@pytest.mark.parametrize("field", [None, "../cpu.onnx", "/cpu.onnx", "missing.onnx"])
def test_optional_cpu_manifest_rejects_invalid_field(tmp_path, field):
    _bundle(tmp_path)
    manifest = json.loads((tmp_path / M.MANIFEST_NAME).read_text())
    manifest["cpu_generator"] = field
    cfg = _pin(tmp_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        M._validate_bundle(cfg)


def test_optional_cpu_manifest_rejects_wrong_source_hash(tmp_path):
    _bundle(tmp_path)
    manifest = json.loads((tmp_path / M.MANIFEST_NAME).read_text())
    (tmp_path / "cpu.onnx").write_bytes(b"wrong source")
    manifest["cpu_generator"] = "cpu.onnx"
    with pytest.raises(ValueError, match="approved source"):
        M._validate_bundle(_pin(tmp_path, manifest))


def test_cpu_natural_duration_session_reuse_and_metadata(tmp_path, models, monkeypatch):
    import sys
    calls = []
    class Session:
        def __init__(self, path, **kw):
            calls.append(("load", kw["providers"]))
        def run(self, outputs, feed):
            assert outputs == ["conv_post"]
            assert feed["x"].shape == (1, 512, 60)
            assert feed["s"].shape == (1, 128)
            assert feed["har"].shape == (1, 22, 3601)
            calls.append(("run",))
            out = np.zeros((1, 22, 3601), np.float32); out[:, 0, :] = .05
            return [out]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session))
    backend = M.KokoroConvOnlyBackend(_cpu_bundle(tmp_path, monkeypatch))
    backend.preload()
    assert calls == []
    backend.frontend.prepare = lambda text, **kw: SimpleNamespace(natural_D=30)
    backend.frontend.render = lambda prepared, target_D: {"T": target_D * 2, "F": target_D * 120 + 1,
        "x": np.zeros((1,512,60), np.float32), "s": np.zeros((1,128), np.float32), "har": np.zeros((1,22,3601), np.float32)}
    for _ in range(2):
        payload, meta = backend.synthesize("OK")
        assert meta["engine"] == "cpu" and meta["fallback"]
        assert meta["natural_D"] == meta["target_D"] == 30 and meta["snap_ratio"] == 0
        assert meta["segments"][0]["source_end"] == 2 and meta["segment_count"] == 1
        assert meta["segments"][0]["fallback_reason"] == "outside_approved_snap"
        assert meta["audio_s"] == .75 and payload.startswith(b"RIFF")
    assert calls == [("load", ["CPUExecutionProvider"]), ("run",), ("run",)]
    assert "prefix" not in models.calls and "tail" not in models.calls
    backend.close()
    assert backend._cpu_generator_session is None


def test_engine_switch_releases_inactive_resources_before_render(tmp_path, models, monkeypatch):
    import sys

    backend = M.KokoroConvOnlyBackend(_cpu_bundle(tmp_path, monkeypatch)); backend.preload()

    class CpuSession:
        def __init__(self, *args, **kwargs): models.calls.append("cpu.load")
        def run(self, *_args, **_kwargs):
            models.calls.append("cpu.run")
            return [np.zeros((1, 22, 3601), np.float32)]
        def close(self): models.calls.append("cpu.close")

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=CpuSession))
    backend._cpu_generator_session = CpuSession()
    models.natural = 160
    models.calls.clear()
    backend.synthesize("NPU request")
    assert models.calls.index("cpu.close") < models.calls.index("render")
    assert backend._cpu_generator_session is None

    backend.prefix_runtime.release_profile_contexts = lambda: models.calls.append("profiles.release")
    backend.frontend.prepare = lambda text, **kw: SimpleNamespace(natural_D=30)
    backend.frontend.render = lambda prepared, target_D: {
        "T": 60, "F": 3601, "x": np.zeros((1, 512, 60), np.float32),
        "s": np.zeros((1, 128), np.float32), "har": np.zeros((1, 22, 3601), np.float32),
        "frontend_detail": {}}
    models.calls.clear()
    backend.synthesize("CPU request")
    assert models.calls.index("profiles.release") < models.calls.index("cpu.load")
    backend.close()


def test_cpu_session_without_close_hook_loses_last_backend_reference():
    import gc
    import weakref

    backend = object.__new__(M.KokoroConvOnlyBackend)
    class Session:
        pass
    session = Session()
    observed = weakref.ref(session)
    backend._cpu_generator_session = session
    del session
    backend._release_cpu_generator()
    gc.collect()
    assert observed() is None and backend._cpu_generator_session is None


def test_cpu_release_error_keeps_owner_and_close_error_is_idempotently_visible(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    class Session:
        def __init__(self): self.calls = 0
        def close(self): self.calls += 1; raise RuntimeError("cpu release failed")
    session = Session(); backend._cpu_generator_session = session
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cpu_generator: cpu release failed"):
            backend.close()
    assert backend._cpu_generator_session is session and session.calls == 1
    assert backend.runtime_info()["cleanup_errors"] == ["cpu_generator: cpu release failed"]


def test_precancelled_request_does_not_evict_inactive_engine(tmp_path, models, monkeypatch):
    backend = M.KokoroConvOnlyBackend(_cpu_bundle(tmp_path, monkeypatch)); backend.preload()
    session = SimpleNamespace(close=lambda: models.calls.append("cpu.close"))
    backend._cpu_generator_session = session
    event = threading.Event(); event.set(); models.natural = 160; models.calls.clear()
    with pytest.raises(M.ConvOnlyCancelled):
        backend.synthesize("cancelled", cancel_event=event)
    assert backend._cpu_generator_session is session and "cpu.close" not in models.calls
    backend.close()


def test_long_stream_plans_lazily_and_never_reprepares_accepted_slice(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    backend.preload()
    prepared_texts = []
    def prepare(text, **kw):
        prepared_texts.append(text)
        if len(text.strip()) > 12: raise M.PhonemeLengthError("too many phonemes")
        return SimpleNamespace(natural_D=160, seed=len(text))
    backend.frontend.prepare = prepare
    text = "Alpha bravo charlie delta echo foxtrot."
    stream = backend.synthesize_stream(text)
    first = next(stream)
    before = list(prepared_texts)
    assert first[1]["source_start"] == 0 and first[1]["source_end"] < len(text)
    assert not any(t.strip() == "foxtrot." for t in before)
    all_chunks = [first, *stream]
    segments = [meta for _, meta in all_chunks if meta["chunk_index"] == 0]
    assert "".join(text[m["source_start"]:m["source_end"]] for m in segments) == text
    assert all(prepared_texts.count(text[m["source_start"]:m["source_end"]]) == 1 for m in segments)
    backend.close()


def test_planner_uses_duration_not_phoneme_limit_and_cancels_between_probes(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    event = threading.Event(); calls = []
    def prepare(text, **kw):
        calls.append(text)
        if len(text) > 1: raise M.PhonemeLengthError("long")
        return SimpleNamespace(natural_D=250, seed=1)
    backend.frontend.prepare = prepare
    payload, meta = backend.synthesize("你好")
    assert len(meta["segments"]) == 2 and all(s["T"] == 480 for s in meta["segments"])
    assert meta["T"] is None and meta["source_end"] == 2
    def cancelled(text, **kw):
        event.set(); return SimpleNamespace(natural_D=250, seed=1)
    backend.frontend.prepare = cancelled
    with pytest.raises(M.ConvOnlyCancelled): backend.synthesize("你好", cancel_event=event)
    backend.close()


def test_real_stream_request_limits_and_cleanup_between_yields(tmp_path, models, monkeypatch):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path))
    with pytest.raises(ValueError, match="10000"):
        next(backend.synthesize_stream("Hi. " * 3000))
    monkeypatch.setattr(M, "MAX_SEGMENTS", 2)
    with pytest.raises(ValueError, match="segments"):
        list(backend.synthesize_stream("One. Two. Three."))
    stream = backend.synthesize_stream("One. Two.")
    next(stream); backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        list(stream)


def test_segmented_total_includes_failed_probe_and_preparation(tmp_path, models):
    import time
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    def prepare(text, **kw):
        time.sleep(.01)
        if len(text) > 1: raise M.PhonemeLengthError("long")
        return SimpleNamespace(natural_D=250, seed=1)
    backend.frontend.prepare = prepare
    _, meta = backend.synthesize("你好")
    assert meta["planning_ms"] >= 30 and meta["total_ms"] >= meta["planning_ms"]
    assert meta["full_rtf"] == pytest.approx(meta["total_ms"] / 1000 / meta["audio_s"])
    assert len(meta["segments"]) == 2 and meta["manifest_sha256"] == backend.config.manifest_sha256
    backend.close()


def test_profile_never_snaps_below_one_frame_per_token(tmp_path, models):
    backend = M.KokoroConvOnlyBackend(_bundle(tmp_path)); backend.preload()
    backend.frontend.prepare = lambda text, **kw: SimpleNamespace(natural_D=202, input_ids=np.zeros((1,202),np.int64))
    _, _, prepared, profile = next(backend._iter_planned_segments("fast speech", "en", 2.0, ()))
    assert prepared.natural_D == 202 and profile["fallback"]
    assert profile["target_D"] >= 202  # CPU will keep the natural 202 frames.
    backend.close()
