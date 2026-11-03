"""Comprehensive unit tests for Kokoro Dual-Bucket TTSBackend (Spec V13).

Validates:
1. Exact exception identity matching server.core.api_execution._is_kokoro_convonly_cancelled.
2. Frontend capacity extension: N_ph = 238 passes, N_ph = 239 raises PhonemeLengthError.
3. Air-defense input validation: empty text, whitespace, or no valid phonetic tokens raises ValueError.
4. Four-quadrant routing decisions (Cases A, B, C, D) and D_out_valid calculation.
5. Deterministic silence padding: speech tokens have 100% natural duration, EOS absorbs Delta D.
6. Manifest SHA256 verification and tamper rejection.
7. End-to-end orchestration and sample-accurate waveform truncation.
8. Factory registration in create_backend and server.core.api_execution is_kokoro check.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import wave

from rkvoice_stream.backends.tts import kokoro_dualbucket as DB
from rkvoice_stream.backends.tts.kokoro_convonly import ConvOnlyCancelled
from rkvoice_stream.backends.tts.kokoro_long32_frontend import (
    KokoroLong32Frontend, PhonemeLengthError, NoPhoneticTokensError, snap_durations,
    FULL_MODEL_SHA256, GENERATOR_SOURCE_SHA256, VOICES, ROUTES,
)
from rkvoice_stream.engine.tts import TTSBackend, create_backend


# =========================================================================
# 1. Exception & Cancelled Class Identity
# =========================================================================

def test_convonly_cancelled_identity():
    """Verify ConvOnlyCancelled has exact module and class identity required by server runtime."""
    exc = ConvOnlyCancelled("render")
    assert type(exc) is ConvOnlyCancelled
    assert exc.__class__.__module__ == "rkvoice_stream.backends.tts.kokoro_convonly"
    assert exc.__class__.__name__ == "ConvOnlyCancelled"

    # Subclasses must NOT match (exact class identity constraint)
    class SubclassCancelled(ConvOnlyCancelled):
        pass

    sub = SubclassCancelled("render")
    assert type(sub) is not ConvOnlyCancelled

    # Foreign exception with same name must NOT match
    class FakeCancelled(RuntimeError):
        pass
    FakeCancelled.__name__ = "ConvOnlyCancelled"
    assert type(FakeCancelled("render")) is not ConvOnlyCancelled


# =========================================================================
# 2. Frontend Capacity Extension (1..238)
# =========================================================================

def test_frontend_ids_capacity_boundary(monkeypatch):
    """N_ph = 238 must pass, N_ph = 239 must raise PhonemeLengthError, empty must raise ValueError."""
    # Build minimal frontend with dummy vocab containing 'a'
    vocab = {" ": 0, "a": 1}
    frontend = object.__new__(KokoroLong32Frontend)
    frontend.vocab = vocab

    # 1. Valid range: N_ph = 238
    phonemes_238 = "a" * 238
    ids = frontend._ids(phonemes_238)
    assert len(ids) == 238
    assert all(i == 1 for i in ids)

    # 2. Exceeds capacity: N_ph = 239 -> PhonemeLengthError
    phonemes_239 = "a" * 239
    with pytest.raises(PhonemeLengthError, match=r"phoneme length outside 1\.\.238: 239"):
        frontend._ids(phonemes_239)

    # 3. Empty phonemes -> ValueError
    with pytest.raises(ValueError, match="text must contain valid phonetic tokens"):
        frontend._ids("")

    # 4. Single phoneme passes
    ids_1 = frontend._ids("a")
    assert len(ids_1) == 1


# =========================================================================
# 3. Empty Input / Air-Defense (空输入防空拦截)
# =========================================================================

@pytest.mark.parametrize("bad_input", ["", "   ", "\t\n\r", "   \t   "])
def test_backend_synthesize_rejects_empty_and_whitespace(bad_input):
    """Empty string and whitespace-only string must raise ValueError('text must contain valid phonetic tokens')."""
    backend = object.__new__(DB.KokoroConvOnlyBackend)
    with pytest.raises(ValueError, match="text must contain valid phonetic tokens"):
        backend.synthesize(bad_input)


def test_backend_synthesize_rejects_non_string():
    """Non-string inputs must raise ValueError."""
    backend = object.__new__(DB.KokoroConvOnlyBackend)
    for bad in (None, 123, [], {}):
        with pytest.raises(ValueError, match="text must contain valid phonetic tokens"):
            backend.synthesize(bad)


# =========================================================================
# 4. Four-Quadrant Routing Decision & D_out_valid
# =========================================================================

def test_route_dualbucket_case_a_padding():
    """Case A: D_nat < 105 -> Bucket 1 (T=240, D_target=120), padding branch, D_out_valid = D_nat."""
    for d_nat in (3, 20, 50, 104):
        res = DB.route_dualbucket(natural_D=d_nat, n_dur=min(d_nat, 30))
        assert res["action"] == "render"
        assert res["bucket"] == "T240"
        assert res["T"] == 240
        assert res["target_D"] == 120
        assert res["branch"] == "padding"
        assert res["D_out_valid"] == d_nat
        assert res["valid_samples"] == 600 * d_nat


def test_route_dualbucket_case_b_snap():
    """Case B: 105 <= D_nat <= 141 and N_dur <= 120 -> Bucket 1, snap branch, D_out_valid = 120."""
    for d_nat in (105, 120, 130, 141):
        res = DB.route_dualbucket(natural_D=d_nat, n_dur=50)
        assert res["action"] == "render"
        assert res["bucket"] == "T240"
        assert res["T"] == 240
        assert res["target_D"] == 120
        assert res["branch"] == "snap"
        assert res["D_out_valid"] == 120
        assert res["valid_samples"] == 600 * 120 == 72000


def test_route_dualbucket_case_c_padding():
    """Case C: 142 <= D_nat < 209 -> Bucket 2 (T=480, D_target=240), padding branch, D_out_valid = D_nat."""
    for d_nat in (142, 175, 200, 208):
        res = DB.route_dualbucket(natural_D=d_nat, n_dur=80)
        assert res["action"] == "render"
        assert res["bucket"] == "T480"
        assert res["T"] == 480
        assert res["target_D"] == 240
        assert res["branch"] == "padding"
        assert res["D_out_valid"] == d_nat
        assert res["valid_samples"] == 600 * d_nat


def test_route_dualbucket_case_c_overflow_into_bucket_2():
    """When N_dur > 120 and D_nat < 209: overflows into Bucket 2, padding branch, D_out_valid = D_nat."""
    # N_dur = 125 cannot fit in Bucket 1 (D_target=120 < 125)
    res = DB.route_dualbucket(natural_D=135, n_dur=125)
    assert res["action"] == "render"
    assert res["bucket"] == "T480"
    assert res["T"] == 480
    assert res["target_D"] == 240
    assert res["branch"] == "padding"
    assert res["D_out_valid"] == 135
    assert res["valid_samples"] == 600 * 135


def test_route_dualbucket_case_d_snap():
    """Case D: 209 <= D_nat <= 282 -> Bucket 2, snap branch, D_out_valid = 240."""
    for d_nat in (209, 240, 260, 282):
        res = DB.route_dualbucket(natural_D=d_nat, n_dur=150)
        assert res["action"] == "render"
        assert res["bucket"] == "T480"
        assert res["T"] == 480
        assert res["target_D"] == 240
        assert res["branch"] == "snap"
        assert res["D_out_valid"] == 240
        assert res["valid_samples"] == 600 * 240 == 144000


def test_route_dualbucket_exceeding_capacity_triggers_split():
    """D_nat > 282 or N_dur > 240 triggers clause splitting action."""
    assert DB.route_dualbucket(natural_D=283, n_dur=100)["action"] == "split"
    assert DB.route_dualbucket(natural_D=200, n_dur=241)["action"] == "split"
    assert DB.route_dualbucket(natural_D=350, n_dur=250)["action"] == "split"


# =========================================================================
# 5. Deterministic Silence Padding Invariant (Zero Deformation)
# =========================================================================

def test_silence_padding_zero_deformation_bucket_1():
    """Speech tokens must retain 100% natural duration, only EOS absorbs Delta D."""
    nat_d = np.array([2, 5, 4, 3, 6], dtype=np.int64)  # sum = 20, len = 5
    target_D = 120
    padded = DB.apply_silence_padding(nat_d, target_D)

    # 1. Exact length preserved
    assert len(padded) == len(nat_d)
    # 2. Zero deformation for all non-EOS tokens
    assert np.array_equal(padded[:-1], nat_d[:-1])
    # 3. EOS absorbs difference exactly
    assert padded[-1] == nat_d[-1] + (target_D - int(nat_d.sum()))
    # 4. Sum matches target exactly
    assert int(padded.sum()) == target_D
    # 5. All durations >= 1
    assert np.all(padded >= 1)


def test_silence_padding_zero_deformation_bucket_2():
    """Bucket 2 silence padding: Delta D = 240 - D_nat absorbed by EOS."""
    np.random.seed(42)
    nat_d = np.random.randint(1, 4, size=80).astype(np.int64)  # D_nat ~ 160
    d_nat = int(nat_d.sum())
    assert 80 <= d_nat < 209
    target_D = 240

    padded = DB.apply_silence_padding(nat_d, target_D)
    assert len(padded) == len(nat_d)
    assert np.array_equal(padded[:-1], nat_d[:-1])
    assert padded[-1] == nat_d[-1] + (target_D - d_nat)
    assert int(padded.sum()) == target_D
    assert np.all(padded >= 1)


def test_silence_padding_rejects_invalid():
    with pytest.raises(ValueError, match="exceeds target_D"):
        DB.apply_silence_padding(np.array([100, 50]), target_D=120)
    with pytest.raises(ValueError, match="durations must be non-empty"):
        DB.apply_silence_padding(np.array([], dtype=np.int64), target_D=120)
    with pytest.raises(ValueError, match="all values >= 1"):
        DB.apply_silence_padding(np.array([1, 0, 2]), target_D=120)


# =========================================================================
# 6. Bundle Creation & Manifest SHA256 Verification
# =========================================================================

def _create_mock_dualbucket_bundle(root: Path, platform: str = "rk3588") -> DB.KokoroDualBucketConfig:
    frontend_dir = root / "frontend"
    frontend_dir.mkdir(parents=True)
    models_dir = root / "models"
    models_dir.mkdir(parents=True)

    # Frontend tokens and models
    (frontend_dir / "tokens.json").write_text(json.dumps({" ": 0, "a": 1}))
    for v in VOICES.values():
        (frontend_dir / f"{v}.bin").write_bytes(b"mock-voice")
    for name in ("frontend_stem.onnx", "duration_head.onnx", "target_tail.onnx"):
        (frontend_dir / name).write_bytes(b"mock-frontend-onnx")

    export_manifest = {
        "schema": "kokoro-v10-generator-export-1",
        "status": "PASS",
        "target": platform,
        "full_model_sha256": FULL_MODEL_SHA256,
        "output_sha256": GENERATOR_SOURCE_SHA256,
    }
    (frontend_dir / "generator-export.manifest.json").write_text(json.dumps(export_manifest))
    export_sha = DB._sha(frontend_dir / "generator-export.manifest.json")

    fm = {
        "schema": "kokoro-v10-long32-frontend-3",
        "status": "PASS",
        "official": True,
        "routes": ROUTES,
        "voices": VOICES,
        "full_source_sha256": FULL_MODEL_SHA256,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "generator_export_manifest_sha256": export_sha,
        "generator_export_manifest": {
            "path": "generator-export.manifest.json",
            "schema": export_manifest["schema"],
            "status": "PASS",
            "sha256": export_sha,
        },
        "stem_model": "frontend_stem.onnx",
        "duration_model": "duration_head.onnx",
        "target_model": "target_tail.onnx",
        "stem_sha256": DB._sha(frontend_dir / "frontend_stem.onnx"),
        "duration_sha256": DB._sha(frontend_dir / "duration_head.onnx"),
        "target_sha256": DB._sha(frontend_dir / "target_tail.onnx"),
        "assets": {p.name: {"bytes": p.stat().st_size, "sha256": DB._sha(p)} for p in frontend_dir.iterdir() if p.suffix != ".onnx"},
    }
    (frontend_dir / "frontend.manifest.json").write_text(json.dumps(fm))

    # Dual-bucket generator models
    (models_dir / "generator_T240.rknn").write_bytes(b"mock-generator-T240-rknn")
    (models_dir / "generator_T480.rknn").write_bytes(b"mock-generator-T480-rknn")

    models_info = {
        "T240": {
            "path": "models/generator_T240.rknn",
            "sha256": DB._sha(models_dir / "generator_T240.rknn"),
            "size": (models_dir / "generator_T240.rknn").stat().st_size,
            "T": 240,
            "D_target": 120,
            "platform": platform,
            "toolkit": "2.3.2",
            "inputs": [[1, 512, 240], [1, 128], [1, 22, 14401]],
            "outputs": [[1, 22, 14401]],
        },
        "T480": {
            "path": "models/generator_T480.rknn",
            "sha256": DB._sha(models_dir / "generator_T480.rknn"),
            "size": (models_dir / "generator_T480.rknn").stat().st_size,
            "T": 480,
            "D_target": 240,
            "platform": platform,
            "toolkit": "2.3.2",
            "inputs": [[1, 512, 480], [1, 128], [1, 22, 28801]],
            "outputs": [[1, 22, 28801]],
        },
    }

    manifest_path = root / DB.MANIFEST_NAME
    files_inv = [
        {"path": p.relative_to(root).as_posix(), "sha256": DB._sha(p), "size": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file() and p != manifest_path
    ]

    manifest = {
        "schema": "kokoro.dualbucket.bundle.v1",
        "platform": platform,
        "precision": "FP16",
        "toolkit": "2.3.2",
        "full_source_sha256": FULL_MODEL_SHA256,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "frontend": "frontend",
        "models": models_info,
        "files": files_inv,
    }
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = DB._sha(manifest_path)
    return DB.KokoroDualBucketConfig(platform, root, manifest_sha)


def test_manifest_sha_mismatch_fails_closed(tmp_path):
    """Tampering with manifest or passing wrong manifest_sha256 must raise ValueError."""
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    bad_cfg = DB.KokoroDualBucketConfig(cfg.platform, cfg.bundle_root, "0" * 64)
    with pytest.raises(ValueError, match="trusted manifest SHA mismatch"):
        DB._validate_dualbucket_bundle(bad_cfg)


def test_manifest_valid_bundle_passes(tmp_path):
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    bundle = DB._validate_dualbucket_bundle(cfg)
    assert bundle["manifest"]["schema"] == "kokoro.dualbucket.bundle.v1"
    assert "T240" in bundle["models"] and "T480" in bundle["models"]


# =========================================================================
# 7. End-to-End Orchestration & Waveform Truncation
# =========================================================================

class MockGeneratorSession:
    def __init__(self, model_path: str):
        self.path = model_path
        self.bucket = "T240" if "T240" in model_path else "T480"

    def run(self, x, s, har):
        # Returns conv_post of shape (1, 22, 60*T + 1)
        cfg = DB.BUCKET_CONFIGS[self.bucket]
        out = np.zeros((1, 22, cfg["F"]), dtype=np.float32)
        out[:, 0, :] = 0.05
        return out


def test_end_to_end_synthesis_cases_and_truncation(tmp_path, monkeypatch):
    """Test full pipeline on simulated cases A, B, C, D verifying output WAV length and metadata."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    # Stub frontend prepare and render to control natural_D
    test_state = SimpleNamespace(natural_D=50, n_tokens=10)

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        toks = np.zeros((1, test_state.n_tokens), dtype=np.int64)
        nat_d = np.ones(test_state.n_tokens, dtype=np.int64)
        nat_d[-1] = test_state.natural_D - (test_state.n_tokens - 1)
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=toks,
            natural_durations=nat_d,
            natural_D=test_state.natural_D,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        t = 2 * target_D
        return {
            "x": np.zeros((1, 512, t), dtype=np.float32),
            "s": np.zeros((1, 128), dtype=np.float32),
            "har": np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
            "T": t,
            "F": 60 * t + 1,
            "route": "a",
            "language": "en-US",
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    # Case A: D_nat = 50 (< 105) -> Bucket 1, padding, truncated to 600 * 50 = 30000 samples
    test_state.natural_D = 50
    test_state.n_tokens = 10
    wav_bytes, meta = backend.synthesize("Hello world.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T240"
    assert meta["branch"] == "padding"
    assert meta["D_out_valid"] == 50
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 50 == 30000

    # Case B: D_nat = 120 (105..141) -> Bucket 1, snap, 600 * 120 = 72000 samples
    test_state.natural_D = 120
    test_state.n_tokens = 30
    wav_bytes, meta = backend.synthesize("Realtime speech synthesis.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T240"
    assert meta["branch"] == "snap"
    assert meta["D_out_valid"] == 120
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 72000

    # Case C: D_nat = 180 (142..208) -> Bucket 2, padding, truncated to 600 * 180 = 108000 samples
    test_state.natural_D = 180
    test_state.n_tokens = 50
    wav_bytes, meta = backend.synthesize("Artificial intelligence at the edge.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T480"
    assert meta["branch"] == "padding"
    assert meta["D_out_valid"] == 180
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 180 == 108000

    # Case D: D_nat = 240 (209..282) -> Bucket 2, snap, 600 * 240 = 144000 samples
    test_state.natural_D = 240
    test_state.n_tokens = 100
    wav_bytes, meta = backend.synthesize("High performance speech recognition and natural sounding audio synthesis.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T480"
    assert meta["branch"] == "snap"
    assert meta["D_out_valid"] == 240
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 144000

    backend.close()


def test_cooperative_cancellation_during_render(tmp_path, monkeypatch):
    """Verify cooperative cancellation raises ConvOnlyCancelled and drains cleanly."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    cancel_evt = threading.Event()
    cancel_evt.set()  # Pre-cancelled

    with pytest.raises(ConvOnlyCancelled) as exc_info:
        backend.synthesize("Hello", language="en-US", voice="af_heart", cancel_event=cancel_evt)
    assert type(exc_info.value) is ConvOnlyCancelled
    assert exc_info.value.__class__.__module__ == "rkvoice_stream.backends.tts.kokoro_convonly"
    backend.close()


# =========================================================================
# 8. Factory Registration & Integration
# =========================================================================

def test_create_backend_kokoro_dualbucket(tmp_path, monkeypatch):
    """Verify create_backend('kokoro_dualbucket') returns KokoroConvOnlyBackend with exact configuration."""
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    monkeypatch.setenv("KOKORO_DUALBUCKET_ROOT", str(tmp_path))
    monkeypatch.setenv("KOKORO_DUALBUCKET_MANIFEST_SHA256", cfg.manifest_sha256)
    monkeypatch.setenv("RK_PLATFORM", "rk3588")

    from rkvoice_stream.backends.tts.kokoro_dualbucket import KokoroConvOnlyBackend

    backend = create_backend("kokoro_dualbucket")
    assert isinstance(backend, KokoroConvOnlyBackend)
    assert backend.name == "kokoro_convonly"
    assert backend.config.bundle_root == tmp_path
    assert backend.config.manifest_sha256 == cfg.manifest_sha256


# =========================================================================
# 9. Additional Major/Minor Tests: Non-Pronouncing Tokens, Eager Preload,
#    Manifest Hardware Contract, Dynamic Cancel & Concurrency Race
# =========================================================================

@pytest.mark.parametrize("invalid_phonemes", [
    "😊🎉",                  # Pure emojis / all unknown
    "...,,,???!",             # Pure punctuation symbols
    "   \t  \n",              # Pure whitespace
    "---——…",                 # Non-speech dashes/ellipses
    "\u200b\u200c\u200d\ufeff", # Invisible zero-width non-phonetic characters
])
def test_non_pronouncing_inputs_raise_valid_phonetic_error(invalid_phonemes):
    """Pure non-pronouncing, pure whitespace, or all-unknown phonemes must raise
    ValueError('text must contain valid phonetic tokens').
    """
    vocab = {" ": 0, "a": 1, ".": 2, ",": 3, "?": 4, "!": 5, "-": 6}
    frontend = object.__new__(KokoroLong32Frontend)
    frontend.vocab = vocab

    with pytest.raises(ValueError, match="text must contain valid phonetic tokens"):
        frontend._ids(invalid_phonemes)


def test_mixed_pronouncing_with_unknown_raises_unknown_dropped():
    """Inputs with valid speech phonemes but some unknown symbols raise unknown/dropped phoneme."""
    vocab = {" ": 0, "h": 1, "e": 2, "l": 3, "o": 4}
    frontend = object.__new__(KokoroLong32Frontend)
    frontend.vocab = vocab

    with pytest.raises(ValueError, match="unknown/dropped phoneme"):
        frontend._ids("hello 😊")


def test_eager_preload_prevents_false_positive_readiness(tmp_path, monkeypatch):
    """If either bucket fails initialization during preload(), backend must NOT report ready."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)

    class FailingSessionGenerator:
        def __init__(self, model_path: str):
            if "T480" in model_path:
                raise RuntimeError("NPU core error initializing T480")

        def run(self, x, s, har):
            return np.zeros((1, 22, 14401), dtype=np.float32)

    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=FailingSessionGenerator)
    assert backend.is_ready() is False

    with pytest.raises(RuntimeError, match="NPU core error initializing T480"):
        backend.preload()

    # Readiness must remain False and cleanup must have completed
    assert backend.is_ready() is False
    assert backend._ready is False
    assert backend._closed is True


def test_rknn_unregistered_failure_instance_released(tmp_path, monkeypatch):
    """If T480 fails during init, both T240 (registered) and T480 (unregistered) must be released."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)

    sessions_created = []

    class TrackedSession:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.released = False
            self.bucket = "T240" if "T240" in model_path else "T480"
            sessions_created.append(self)

        def init_runtime(self):
            if self.bucket == "T480":
                raise RuntimeError("NPU core error initializing T480")
            return 0

        def release(self):
            self.released = True

        def close(self):
            self.released = True

        def run(self, x, s, har):
            return np.zeros((1, 22, 14401), dtype=np.float32)

    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=TrackedSession)
    assert backend.is_ready() is False

    with pytest.raises(RuntimeError, match="NPU core error initializing T480"):
        backend.preload()

    # Must have attempted to create both sessions
    assert len(sessions_created) == 2
    t240 = next(s for s in sessions_created if s.bucket == "T240")
    t480 = next(s for s in sessions_created if s.bucket == "T480")

    # Both the registered T240 and the failed-during-init T480 must be released!
    assert t240.released is True
    assert t480.released is True
    assert backend.is_ready() is False
    assert backend._ready is False
    assert backend._closed is True


@pytest.mark.parametrize("kind", ["compiler", "platform", "inputs", "outputs"])
def test_manifest_platform_and_tensor_dimensions_gates(tmp_path, kind):
    """Manifest must assert platform, compiler=='2.3.2', and exact tensor dimensions."""
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    manifest_path = tmp_path / DB.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())

    if kind == "compiler":
        manifest["compiler"] = "2.3.1"
        manifest["models"]["T240"]["compiler"] = "2.3.1"
    elif kind == "platform":
        manifest["platform"] = "rk3576"  # config is rk3588
        manifest["models"]["T240"]["platform"] = "rk3576"
    elif kind == "inputs":
        manifest["models"]["T240"]["inputs"] = [[1, 512, 120], [1, 128], [1, 22, 14401]]  # wrong shape
    elif kind == "outputs":
        manifest["models"]["T480"]["outputs"] = [[1, 22, 14401]]  # wrong output for T480

    manifest_path.write_text(json.dumps(manifest))
    new_sha = DB._sha(manifest_path)
    test_cfg = DB.KokoroDualBucketConfig("rk3588", tmp_path, new_sha)

    with pytest.raises(ValueError, match="(mismatch|expected)"):
        DB._validate_dualbucket_bundle(test_cfg)


def test_mid_run_dynamic_cancellation(tmp_path, monkeypatch):
    """Test dynamic cancellation triggered mid-stream inside the generator execution."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)

    cancel_evt = threading.Event()

    class MidRunCancellingGenerator:
        def __init__(self, model_path: str):
            pass

        def run(self, x, s, har):
            # Dynamically fire cancel mid-run
            cancel_evt.set()
            out = np.zeros((1, 22, 14401), dtype=np.float32)
            out[:, 0, :] = 0.05
            return out

    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MidRunCancellingGenerator)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 10), dtype=np.int64),
            natural_durations=np.full(10, 5, dtype=np.int64),
            natural_D=50,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        t = 2 * target_D
        return {
            "x": np.zeros((1, 512, t), dtype=np.float32),
            "s": np.zeros((1, 128), dtype=np.float32),
            "har": np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
            "T": t,
            "F": 60 * t + 1,
            "route": "a",
            "language": "en-US",
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    # Synthesize should detect cancel_evt right after generator returns
    with pytest.raises(ConvOnlyCancelled) as exc_info:
        backend.synthesize("Hello world.", language="en-US", voice="af_heart", cancel_event=cancel_evt)

    assert type(exc_info.value) is ConvOnlyCancelled
    assert exc_info.value.__class__.__module__ == "rkvoice_stream.backends.tts.kokoro_convonly"
    backend.close()


def test_concurrent_synthesis_and_close_race(tmp_path, monkeypatch):
    """Concurrent synthesis requests while closing backend must serialize cleanly without deadlocks."""
    import concurrent.futures
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 10), dtype=np.int64),
            natural_durations=np.full(10, 5, dtype=np.int64),
            natural_D=50,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        t = 2 * target_D
        return {
            "x": np.zeros((1, 512, t), dtype=np.float32),
            "s": np.zeros((1, 128), dtype=np.float32),
            "har": np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
            "T": t,
            "F": 60 * t + 1,
            "route": "a",
            "language": "en-US",
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    results = []
    errors = []
    active_event = threading.Event()

    def worker(idx: int):
        try:
            cancel = threading.Event() if idx == 1 else None
            if cancel:
                cancel.set()
            wav_bytes, meta = backend.synthesize(
                "Concurrent test.", language="en-US", voice="af_heart", cancel_event=cancel
            )
            results.append(len(wav_bytes))
            active_event.set()
        except BaseException as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = [executor.submit(worker, i) for i in range(8)]
        # Ensure at least one worker has entered and successfully completed synthesis
        assert active_event.wait(timeout=2.0)
        assert len(results) >= 1
        # Concurrently close the backend while remaining workers are queued or in flight
        backend.close()
        for f in futs:
            f.result()

    assert len(results) >= 1
    assert len(errors) >= 1
    assert len(results) + len(errors) == 8

    for err in errors:
        assert isinstance(err, (RuntimeError, ConvOnlyCancelled)), f"Unexpected error type: {type(err)}"
        msg = str(err)
        assert (
            "backend is closed" in msg
            or "Conv-only synthesis cancelled" in msg
            or "cancelled" in msg.lower()
        ), f"Unexpected error message: {msg}"

    assert backend.is_ready() is False
    assert backend._closed is True


# =========================================================================
# 10. Fail-Closed CPU Generator, Stream Punctuation Filtering & Partitioning
# =========================================================================

def test_cpu_generator_fail_closed_by_default(tmp_path, monkeypatch):
    """When RKNNLite is missing, CPU generator must fail closed unless allow_cpu_generator=True."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "rknnlite" in name:
            raise ImportError("No module named 'rknnlite'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg)
    assert backend.config.allow_cpu_generator is False

    with pytest.raises(RuntimeError, match="RKNN runtime not available \\(rknnlite missing or load failed\\)"):
        backend.preload()


def test_cpu_generator_allowed_when_explicitly_configured(tmp_path, monkeypatch):
    """When allow_cpu_generator=True and cpu_generator exists, fallback is permitted."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    _create_mock_dualbucket_bundle(tmp_path)

    # Create dummy cpu_generator onnx file and update manifest
    cpu_gen_path = tmp_path / "generator.onnx"
    cpu_gen_path.write_bytes(b"dummy onnx")

    manifest_path = tmp_path / DB.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["cpu_generator"] = "generator.onnx"
    manifest["files"].append({
        "path": "generator.onnx",
        "size": len(b"dummy onnx"),
        "sha256": hashlib.sha256(b"dummy onnx").hexdigest(),
    })
    manifest_path.write_text(json.dumps(manifest))
    new_sha = DB._sha(manifest_path)

    cfg = DB.KokoroDualBucketConfig("rk3588", tmp_path, new_sha, allow_cpu_generator=True)

    class MockOrtSession:
        def __init__(self, path, providers=None):
            self.path = path

        def run(self, out_names, inputs):
            cfg = DB.BUCKET_CONFIGS["T240"]
            out = np.zeros((1, 22, cfg["F"]), dtype=np.float32)
            out[:, 0, :] = 0.05
            return [out]

        def get_inputs(self):
            return ["x", "s", "har"]

    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", MockOrtSession)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "rknnlite" in name:
            raise ImportError("No module named 'rknnlite'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = DB.KokoroConvOnlyBackend(cfg)
    backend.preload()
    assert backend.is_ready() is True
    backend.close()


def test_stream_synthesize_skips_punctuation_only_segments(tmp_path, monkeypatch):
    """Stream synthesis on 'Hello world! !!' must skip '!!' without raising ValueError."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 10), dtype=np.int64),
            natural_durations=np.full(10, 5, dtype=np.int64),
            natural_D=50,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        t = 2 * target_D
        return {
            "x": np.zeros((1, 512, t), dtype=np.float32),
            "s": np.zeros((1, 128), dtype=np.float32),
            "har": np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
            "T": t,
            "F": 60 * t + 1,
            "route": "a",
            "language": "en-US",
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    chunks = list(backend.synthesize_stream("Hello world! !!", language="en-US", voice="af_heart"))
    assert len(chunks) > 0
    # Every chunk generated must correspond to 'Hello world! '
    for pcm, meta in chunks:
        assert meta["sentence_text"] == "Hello world! "

    wav_bytes, meta = backend.synthesize("Hello world! !!", language="en-US", voice="af_heart")
    assert len(wav_bytes) > 0
    assert meta["segment_count"] == 1
    backend.close()


def test_long_sentence_recursive_partitioning(tmp_path, monkeypatch):
    """A sentence exceeding 282 natural_D must be partitioned cleanly without crashing."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        d_val = min(300, max(20, len(text) * 4))
        n_tok = min(150, max(5, len(text)))
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, n_tok), dtype=np.int64),
            natural_durations=np.full(n_tok, d_val // n_tok, dtype=np.int64),
            natural_D=d_val,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        t = 2 * target_D
        return {
            "x": np.zeros((1, 512, t), dtype=np.float32),
            "s": np.zeros((1, 128), dtype=np.float32),
            "har": np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
            "T": t,
            "F": 60 * t + 1,
            "route": "a",
            "language": "en-US",
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    long_clause = "This is a very long sentence that needs to be partitioned because it exceeds the maximum dual-bucket capacity."
    wav_bytes, meta = backend.synthesize(long_clause, language="en-US", voice="af_heart")
    assert len(wav_bytes) > 0
    assert meta["segmented"] is True
    assert meta["segment_count"] >= 2
    backend.close()


def test_all_segments_non_pronouncing_raises_no_phonetic_tokens(tmp_path, monkeypatch):
    """When all segments produce no valid phonetic tokens, synthesize and synthesize_stream must raise NoPhoneticTokensError."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    # Case 1: text with only whitespace and punctuation
    for bad_text in ("   ", "...,,,", "---——…", "   \t  \n"):
        with pytest.raises(NoPhoneticTokensError, match="text must contain valid phonetic tokens"):
            backend.synthesize(bad_text, language="en-US", voice="af_heart")
        with pytest.raises(NoPhoneticTokensError, match="text must contain valid phonetic tokens"):
            list(backend.synthesize_stream(bad_text, language="en-US", voice="af_heart"))

    # Case 2: fake_prepare always raises NoPhoneticTokensError (e.g. all tokens dropped/unknown)
    def failing_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        raise NoPhoneticTokensError("text must contain valid phonetic tokens")

    backend.frontend.prepare = failing_prepare
    with pytest.raises(NoPhoneticTokensError, match="text must contain valid phonetic tokens"):
        backend.synthesize("Some text.", language="en-US", voice="af_heart")
    with pytest.raises(NoPhoneticTokensError, match="text must contain valid phonetic tokens"):
        list(backend.synthesize_stream("Some text.", language="en-US", voice="af_heart"))

    backend.close()

