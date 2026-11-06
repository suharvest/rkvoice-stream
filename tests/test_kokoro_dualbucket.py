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
import re
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
    VoiceWarmupState, VOICE_SHAPE,
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
        if "T50" in model_path:
            self.bucket = "T50"
        elif "T240" in model_path:
            self.bucket = "T240"
        else:
            self.bucket = "T480"

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


# =========================================================================
# 9. T50 Windowed Generator & Overlap-Add Crossfade (Decoder Windowing)
# =========================================================================

def test_depop_append_crossfade_properties():
    """Verify depop_append properties: linear symmetry, exact length, edge clamping."""
    # 1. Linear symmetry: weights sum to 1.0 everywhere
    overlap = 480
    fade_in = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
    fade_out = 1.0 - fade_in
    assert np.allclose(fade_in + fade_out, 1.0)
    assert np.isclose(fade_in[0], 0.5 / 480)
    assert np.isclose(fade_in[-1], 1.0 - 0.5 / 480)

    # 2. Perfect continuous ramp blending
    a = np.arange(1000, dtype=np.float32)
    b = np.arange(1000 - overlap, 2000 - overlap, dtype=np.float32)
    stitched = DB.depop_append(a, b, crossfade_samples=overlap)
    assert stitched.shape == (2000 - overlap,)
    expected = np.arange(2000 - overlap, dtype=np.float32)
    assert np.allclose(stitched, expected, atol=1e-4)

    # 3. Edge cases
    empty = np.empty((0,), dtype=np.float32)
    chunk = np.ones(500, dtype=np.float32)
    assert np.array_equal(DB.depop_append(empty, chunk), chunk)
    assert np.array_equal(DB.depop_append(chunk, empty), chunk)

    # Small chunks smaller than 480
    small_a = np.full(50, 1.0, dtype=np.float32)
    small_b = np.full(50, 2.0, dtype=np.float32)
    res = DB.depop_append(small_a, small_b, crossfade_samples=480)
    assert res.shape == (50,)  # overlap clamped to min(480, 50, 50) = 50


def test_generator_t50_windowed_slicing_and_padding(tmp_path):
    """Verify DualBucketGeneratorRunner.run_windowed handles sequences from T=5 to T=500."""
    models = {"T50": tmp_path / "generator_T50.rknn"}
    models_info = {
        "T50": {
            "path": "generator_T50.rknn",
            "T": 50,
            "inputs": [[1, 512, 50], [1, 128], [1, 22, 3001]],
            "outputs": [[1, 22, 3001]],
        }
    }
    models["T50"].write_bytes(b"dummy")

    class WindowMockSession:
        def __init__(self, path):
            self.calls = []

        def run(self, x, s, har):
            assert x.shape == (1, 512, 50)
            assert s.shape == (1, 128)
            assert har.shape == (1, 22, 3001)
            self.calls.append((x.shape, har.shape))
            out = np.zeros((1, 22, 3001), dtype=np.float32)
            out[:, 0, :] = 0.05
            return out

    runner = DB.DualBucketGeneratorRunner(
        models=models,
        models_info=models_info,
        platform="rk3588",
        runtime_cls=WindowMockSession,
    )
    assert runner.has_t50 is True

    # 1. Short sequence: T = 20 (< 50) -> 1 window
    x_short = np.zeros((1, 512, 20), dtype=np.float32)
    har_short = np.zeros((1, 22, 60 * 20 + 1), dtype=np.float32)
    s = np.zeros((1, 128), dtype=np.float32)
    wav, gen_ms, istft_ms = runner.run_windowed(x_short, s, har_short)
    assert wav.shape == (15000,)
    assert gen_ms >= 0 and istft_ms >= 0

    # 2. Exact sequence: T = 50 -> 1 window
    x_50 = np.zeros((1, 512, 50), dtype=np.float32)
    har_50 = np.zeros((1, 22, 3001), dtype=np.float32)
    wav, _, _ = runner.run_windowed(x_50, s, har_50)
    assert wav.shape == (15000,)

    # 3. Two-window sequence: T = 70 -> 2 windows (acc_size = 28800)
    x_70 = np.zeros((1, 512, 70), dtype=np.float32)
    har_70 = np.zeros((1, 22, 60 * 70 + 1), dtype=np.float32)
    wav, _, _ = runner.run_windowed(x_70, s, har_70)
    assert wav.shape == (28800,)

    # 4. Long sequence: T = 480 -> 11 windows (acc_size = 153000)
    x_480 = np.zeros((1, 512, 480), dtype=np.float32)
    har_480 = np.zeros((1, 22, 60 * 480 + 1), dtype=np.float32)
    wav, _, _ = runner.run_windowed(x_480, s, har_480)
    assert wav.shape == (153000,)


def _create_mock_t50_bundle(root: Path, platform: str = "rk3588", include_dual: bool = False) -> DB.KokoroDualBucketConfig:
    frontend_dir = root / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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

    (models_dir / "generator_T50.rknn").write_bytes(b"mock-generator-T50-rknn")
    models_info = {
        "T50": {
            "path": "models/generator_T50.rknn",
            "sha256": DB._sha(models_dir / "generator_T50.rknn"),
            "size": (models_dir / "generator_T50.rknn").stat().st_size,
            "T": 50,
            "D_target": 25,
            "platform": platform,
            "toolkit": "2.3.2",
            "inputs": [[1, 512, 50], [1, 128], [1, 22, 3001]],
            "outputs": [[1, 22, 3001]],
        }
    }
    if include_dual:
        (models_dir / "generator_T240.rknn").write_bytes(b"mock-generator-T240-rknn")
        (models_dir / "generator_T480.rknn").write_bytes(b"mock-generator-T480-rknn")
        models_info["T240"] = {
            "path": "models/generator_T240.rknn",
            "sha256": DB._sha(models_dir / "generator_T240.rknn"),
            "size": (models_dir / "generator_T240.rknn").stat().st_size,
            "T": 240,
            "D_target": 120,
            "platform": platform,
            "toolkit": "2.3.2",
            "inputs": [[1, 512, 240], [1, 128], [1, 22, 14401]],
            "outputs": [[1, 22, 14401]],
        }
        models_info["T480"] = {
            "path": "models/generator_T480.rknn",
            "sha256": DB._sha(models_dir / "generator_T480.rknn"),
            "size": (models_dir / "generator_T480.rknn").stat().st_size,
            "T": 480,
            "D_target": 240,
            "platform": platform,
            "toolkit": "2.3.2",
            "inputs": [[1, 512, 480], [1, 128], [1, 22, 28801]],
            "outputs": [[1, 22, 28801]],
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


def test_t50_manifest_validation(tmp_path):
    """Verify validation of bundles with T50 model and rejects invalid configurations."""
    cfg = _create_mock_t50_bundle(tmp_path)
    bundle = DB._validate_dualbucket_bundle(cfg)
    assert "T50" in bundle["models"]
    assert bundle["manifest"]["schema"] == "kokoro.dualbucket.bundle.v1"


def test_t50_end_to_end_synthesis_short_medium_long(tmp_path, monkeypatch):
    """Full end-to-end synthesis with T50 windowed generator across short, medium and long sentences."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()
    assert backend.is_ready() is True
    assert backend.generator.has_t50 is True

    test_state = SimpleNamespace(natural_D=20, n_tokens=5)

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

    # 1. Short sentence: D_nat = 20 -> valid_samples = 600 * 20 = 12000
    test_state.natural_D = 20
    test_state.n_tokens = 5
    wav_bytes, meta = backend.synthesize("Hi.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T50"
    assert meta["branch"] == "windowed"
    assert meta["D_out_valid"] == 20
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 20 == 12000

    # 2. Medium sentence: D_nat = 80 -> valid_samples = 600 * 80 = 48000
    test_state.natural_D = 80
    test_state.n_tokens = 20
    wav_bytes, meta = backend.synthesize("Speech synthesis on RKNN NPU.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T50"
    assert meta["D_out_valid"] == 80
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 80 == 48000

    # 3. Long sentence: D_nat = 240 -> valid_samples = 600 * 240 = 144000
    test_state.natural_D = 240
    test_state.n_tokens = 60
    wav_bytes, meta = backend.synthesize("Extremely long utterance across many T50 slices.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T50"
    assert meta["D_out_valid"] == 240
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 240 == 144000

    # 4. Super-long sentence: D_nat = 350 -> valid_samples = 600 * 350 = 210000 (rendered in 1 piece via T50 windowing!)
    test_state.natural_D = 350
    test_state.n_tokens = 90
    wav_bytes, meta = backend.synthesize("Super long clause that would exceed dual-bucket limit without splitting.", language="en-US", voice="af_heart")
    assert meta["selected_bucket"] == "T50"
    assert meta["D_out_valid"] == 350
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 600 * 350 == 210000

    backend.close()


def test_t50_cooperative_cancellation_during_windowing(tmp_path, monkeypatch):
    """Verify cooperative cancellation stops multi-slice windowing immediately."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    cancel_evt = threading.Event()

    class CancelTriggerSession(MockGeneratorSession):
        def __init__(self, path):
            super().__init__(path)
            self.count = 0

        def run(self, x, s, har):
            self.count += 1
            if self.count >= 2:
                cancel_evt.set()
            return super().run(x, s, har)

    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=CancelTriggerSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 10), dtype=np.int64),
            natural_durations=np.full(10, 20, dtype=np.int64),
            natural_D=200,
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

    with pytest.raises(ConvOnlyCancelled) as exc_info:
        backend.synthesize("Long sentence to cancel.", language="en-US", voice="af_heart", cancel_event=cancel_evt)

    assert exc_info.value.stage == "generator"
    backend.close()


def test_t50_streaming_synthesis(tmp_path, monkeypatch):
    """Verify streaming synthesis with T50 windowed generator yields expected chunks."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, 10, dtype=np.int64),
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

    chunks = list(backend.synthesize_stream("Hello streaming world.", language="en-US", voice="af_heart"))
    assert len(chunks) > 0
    total_samples = sum(c[0].shape[0] for c in chunks)
    assert total_samples == 600 * 50 == 30000
    assert chunks[0][1]["selected_bucket"] == "T50"
    backend.close()


def test_t50_depop_spectral_continuity():
    """Verify 480-sample symmetric crossfade eliminates pops and achieves continuous harmonic synthesis."""
    sr = 24000
    freq = 440.0
    t0 = np.arange(15000) / sr
    w0 = np.sin(2 * np.pi * freq * t0).astype(np.float32)

    t1 = (np.arange(15000) + 13800) / sr
    w1 = np.sin(2 * np.pi * freq * t1).astype(np.float32)

    chunk0 = w0[:14640]
    chunk1 = w1[360:]
    stitched = DB.depop_append(chunk0, chunk1, crossfade_samples=480)

    t_full = np.arange(stitched.size) / sr
    ideal = np.sin(2 * np.pi * freq * t_full).astype(np.float32)

    max_err = np.max(np.abs(stitched - ideal))
    assert max_err < 1e-4


# =========================================================================
# 10. Streaming & Latency Optimizations (Windowed Streaming, Pipelining, Affinity)
# =========================================================================

@pytest.mark.parametrize("T_len", [20, 50, 70, 150, 240, 480])
@pytest.mark.parametrize("trunc_mode", ["none", "exact", "partial"])
def test_windowed_stream_numerical_equivalence_to_run_windowed(tmp_path, T_len, trunc_mode):
    """Verify run_windowed_stream output concatenated is bit-exact identical to run_windowed."""
    models = {"T50": tmp_path / "generator_T50.rknn"}
    models_info = {
        "T50": {
            "path": "generator_T50.rknn",
            "T": 50,
            "inputs": [[1, 512, 50], [1, 128], [1, 22, 3001]],
            "outputs": [[1, 22, 3001]],
        }
    }
    models["T50"].write_bytes(b"dummy")

    class UniqueWaveSession:
        def __init__(self, path):
            self.count = 0

        def run(self, x, s, har):
            self.count += 1
            out = np.zeros((1, 22, 3001), dtype=np.float32)
            out[:, 0, :] = 0.05 * self.count
            out[:, 1, :] = 0.02 * np.sin(np.linspace(0, np.pi, 3001))
            return out

    runner = DB.DualBucketGeneratorRunner(
        models=models,
        models_info=models_info,
        platform="rk3588",
        runtime_cls=UniqueWaveSession,
    )

    x = np.zeros((1, 512, T_len), dtype=np.float32)
    s = np.zeros((1, 128), dtype=np.float32)
    har = np.zeros((1, 22, 60 * T_len + 1), dtype=np.float32)

    ref_wav, _, _ = runner.run_windowed(x, s, har)

    if trunc_mode == "none":
        vs = None
        expected = ref_wav
    elif trunc_mode == "exact":
        vs = 600 * (T_len // 2)
        expected = ref_wav[:vs]
    else:
        vs = min(ref_wav.size, 14200)
        expected = ref_wav[:vs]

    runner_stream = DB.DualBucketGeneratorRunner(
        models=models,
        models_info=models_info,
        platform="rk3588",
        runtime_cls=UniqueWaveSession,
    )
    chunks = list(runner_stream.run_windowed_stream(x, s, har, valid_samples=vs))
    streamed_wav = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)

    assert streamed_wav.shape == expected.shape
    assert np.array_equal(streamed_wav, expected)
    assert streamed_wav.dtype == np.float32


def test_synthesize_stream_window_level_chunks_and_metadata(tmp_path, monkeypatch):
    """Verify synthesize_stream yields 40ms PCM chunks with full stream metadata as each window finishes."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    test_d = 30
    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, test_d // 5, dtype=np.int64),
            natural_D=test_d,
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

    chunks = list(backend.synthesize_stream("This is a streaming test sentence.", language="en-US", voice="af_heart"))
    assert len(chunks) > 0

    total_samples = 0
    expected_total_samples = 600 * test_d
    chunk_size_40ms = int(24000 * 0.04)

    for i, (pcm, meta) in enumerate(chunks):
        assert isinstance(pcm, np.ndarray)
        assert pcm.dtype == np.float32
        total_samples += pcm.size

        assert meta["chunk_index"] == i
        assert meta["streaming_mode"] == "sentence"
        assert meta["supports_streaming"] is True
        assert meta["sample_rate"] == 24000
        assert meta["selected_bucket"] == "T50"
        assert meta["sentence_text"] == "This is a streaming test sentence."
        assert meta["sentence_index"] == 0
        assert "audio_s" in meta
        assert "peak" in meta
        assert meta["finite"] is True

        if i < len(chunks) - 1:
            assert meta["segment_complete"] is False
            assert pcm.size == chunk_size_40ms
        else:
            assert meta["segment_complete"] is True

    assert total_samples == expected_total_samples
    backend.close()


def test_pipelined_multisentence_streaming(tmp_path, monkeypatch):
    """Verify multi-sentence asynchronous pipelining renders sentence i+1 while sentence i streams."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    render_calls = []

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, 4, dtype=np.int64),
            natural_D=20,
            style=np.zeros((1, 128), dtype=np.float32),
            speed=np.array([1.0], dtype=np.float32),
            stem_text=None,
            stem_style=None,
            timings={},
        )

    def fake_render(prepared, target_D, durations):
        render_calls.append(prepared.text)
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

    text = "First sentence. Second sentence! Third sentence."
    stream = backend.synthesize_stream(text, language="en-US", voice="af_heart")

    first_chunk = next(stream)
    assert first_chunk[1]["sentence_index"] == 0
    assert first_chunk[1]["sentence_text"] == "First sentence. "

    all_chunks = [first_chunk, *list(stream)]
    assert len(all_chunks) > 0

    sentences_seen = sorted(set(meta["sentence_index"] for _, meta in all_chunks))
    assert sentences_seen == [0, 1, 2]

    for pcm, meta in all_chunks:
        if meta["sentence_index"] == 0:
            assert meta["sentence_text"] == "First sentence. "
        elif meta["sentence_index"] == 1:
            assert meta["sentence_text"] == "Second sentence! "
        elif meta["sentence_index"] == 2:
            assert meta["sentence_text"] == "Third sentence."

    backend.close()


def test_pipelined_streaming_cancellation_and_empty_sentences(tmp_path, monkeypatch):
    """Verify cancellation aborts background pipeline and empty sentences are cleanly skipped."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroConvOnlyBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        return SimpleNamespace(
            route="a",
            language="en-US",
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, 4, dtype=np.int64),
            natural_D=20,
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

    # 1. Empty sentence interspersed
    text_with_empty = "First sentence. ... ??? Second sentence!"
    chunks = list(backend.synthesize_stream(text_with_empty, language="en-US", voice="af_heart"))
    assert len(chunks) > 0
    sentences = set(meta["sentence_text"].strip() for _, meta in chunks)
    assert "First sentence." in sentences
    assert "Second sentence!" in sentences
    assert "..." not in sentences
    assert "???" not in sentences

    # 2. Cancellation during streaming
    cancel_evt = threading.Event()
    stream = backend.synthesize_stream("Sentence one. Sentence two. Sentence three.", language="en-US", voice="af_heart", cancel_event=cancel_evt)
    _ = next(stream)
    cancel_evt.set()

    with pytest.raises(ConvOnlyCancelled):
        list(stream)

    backend.close()


def test_set_big_core_affinity_helper(monkeypatch):
    """Verify set_big_core_affinity handles Linux affinity checks and falls back gracefully."""
    recorded_affinity = []

    # Case 1: 8+ cores available including {4, 5, 6, 7}
    def mock_getaffinity(pid):
        return {0, 1, 2, 3, 4, 5, 6, 7}

    def mock_setaffinity(pid, mask):
        recorded_affinity.append(set(mask))

    monkeypatch.setattr(os, "sched_getaffinity", mock_getaffinity, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", mock_setaffinity, raising=False)

    res = DB.set_big_core_affinity()
    assert res is True
    assert recorded_affinity == [{4, 5, 6, 7}]

    # Case 2: Fewer than 8 cores / cores 4-7 not in set
    recorded_affinity.clear()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    res = DB.set_big_core_affinity()
    assert res is False
    assert len(recorded_affinity) == 0

    # Case 3: OSError / permission error in container
    def mock_failing_set(pid, mask):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "sched_getaffinity", mock_getaffinity, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", mock_failing_set, raising=False)
    res = DB.set_big_core_affinity()
    assert res is False

    # Case 4: Non-Linux / sched_getaffinity missing
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    res = DB.set_big_core_affinity()
    assert res is False


def test_target_tail_session_and_graph_optimizations(monkeypatch):
    """Verify _session_options_factory enables ORT_ENABLE_ALL, ORT_SEQUENTIAL and 4 big cores default."""
    import onnxruntime as ort

    # Case 1: Default intra when not set in env -> 4
    monkeypatch.delenv("KOKORO_FRONTEND_INTRA_OP_THREADS", raising=False)
    monkeypatch.delenv("KOKORO_FRONTEND_INTER_OP_THREADS", raising=False)

    opts = KokoroLong32Frontend._session_options_factory(None, None)
    assert opts is not None
    assert opts.intra_op_num_threads == 4
    assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL

    # Case 2: Explicit intra
    opts_custom = KokoroLong32Frontend._session_options_factory(6, 2)
    assert opts_custom.intra_op_num_threads == 6
    assert opts_custom.inter_op_num_threads == 2
    assert opts_custom.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert opts_custom.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL

    # Case 3: Env var override
    monkeypatch.setenv("KOKORO_FRONTEND_INTRA_OP_THREADS", "8")
    opts_env = KokoroLong32Frontend._session_options_factory(None, None)
    assert opts_env.intra_op_num_threads == 8


# =========================================================================
# Clause-Level Streaming Split & Pipeline Tests
# =========================================================================

def test_clause_streaming_split_and_pipeline(tmp_path, monkeypatch):
    """Verify clause-level splitting and async pipelining in KokoroDualBucketBackend."""
    # 1. Pattern level verification
    text_en = "First clause, second clause. Third sentence."
    text_zh = "今天天气真好，阳光明媚。"
    text_num = "Count is 1,000, and value is 3.14."

    # Clause pattern splits clauses and sentences
    raw_en_c = re.findall(DB.SPLIT_CLAUSE_PATTERN, text_en, flags=re.S)
    assert raw_en_c == ["First clause, ", "second clause. ", "Third sentence."]
    assert "".join(raw_en_c) == text_en

    raw_zh_c = re.findall(DB.SPLIT_CLAUSE_PATTERN, text_zh, flags=re.S)
    assert raw_zh_c == ["今天天气真好，", "阳光明媚。"]
    assert "".join(raw_zh_c) == text_zh

    # Sentence pattern only splits full sentences
    raw_en_s = re.findall(DB.SPLIT_SENTENCE_PATTERN, text_en, flags=re.S)
    assert raw_en_s == ["First clause, second clause. ", "Third sentence."]
    assert "".join(raw_en_s) == text_en

    raw_zh_s = re.findall(DB.SPLIT_SENTENCE_PATTERN, text_zh, flags=re.S)
    assert raw_zh_s == ["今天天气真好，阳光明媚。"]
    assert "".join(raw_zh_s) == text_zh

    # Number / decimal handling: numbers must NOT be split inside 1,000 or 3.14
    raw_num_c = re.findall(DB.SPLIT_CLAUSE_PATTERN, text_num, flags=re.S)
    assert raw_num_c == ["Count is 1,000, ", "and value is 3.14."]
    assert "".join(raw_num_c) == text_num

    raw_num_s = re.findall(DB.SPLIT_SENTENCE_PATTERN, text_num, flags=re.S)
    assert raw_num_s == ["Count is 1,000, and value is 3.14."]
    assert "".join(raw_num_s) == text_num

    # 2. Pipeline execution & metadata tracking with mock backend
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroDualBucketBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        route = ROUTES.get(language, "a")
        return SimpleNamespace(
            route=route,
            language=language,
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, 4, dtype=np.int64),
            natural_D=20,
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
            "route": prepared.route,
            "language": prepared.language,
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    # Test synthesize_stream with clause_split=True
    chunks_c = list(backend.synthesize_stream(text_en, language="en-US", voice="af_heart", clause_split=True))
    assert len(chunks_c) > 0
    indices_c = sorted(set(m["sentence_index"] for _, m in chunks_c))
    assert indices_c == [0, 1, 2]
    expected_texts_c = {0: "First clause, ", 1: "second clause. ", 2: "Third sentence."}
    for _, meta in chunks_c:
        assert meta["sentence_text"] == expected_texts_c[meta["sentence_index"]]
        assert meta["streaming_mode"] == "sentence"
        assert meta["supports_streaming"] is True

    # Test synthesize_stream with clause_split=False
    chunks_s = list(backend.synthesize_stream(text_en, language="en-US", voice="af_heart", clause_split=False))
    assert len(chunks_s) > 0
    indices_s = sorted(set(m["sentence_index"] for _, m in chunks_s))
    assert indices_s == [0, 1]
    expected_texts_s = {0: "First clause, second clause. ", 1: "Third sentence."}
    for _, meta in chunks_s:
        assert meta["sentence_text"] == expected_texts_s[meta["sentence_index"]]

    # Test synthesize_stream with Chinese text and clause_split=True
    chunks_zh = list(backend.synthesize_stream(text_zh, language="zh", voice="zf_xiaobei", clause_split=True))
    assert len(chunks_zh) > 0
    indices_zh = sorted(set(m["sentence_index"] for _, m in chunks_zh))
    assert indices_zh == [0, 1]
    expected_texts_zh = {0: "今天天气真好，", 1: "阳光明媚。"}
    for _, meta in chunks_zh:
        assert meta["sentence_text"] == expected_texts_zh[meta["sentence_index"]]

    # Test KOKORO_STREAM_CLAUSE_SPLIT environment variable defaults
    monkeypatch.setenv("KOKORO_STREAM_CLAUSE_SPLIT", "0")
    chunks_env0 = list(backend.synthesize_stream(text_en, language="en-US", voice="af_heart"))
    assert sorted(set(m["sentence_index"] for _, m in chunks_env0)) == [0, 1]

    monkeypatch.setenv("KOKORO_STREAM_CLAUSE_SPLIT", "1")
    chunks_env1 = list(backend.synthesize_stream(text_en, language="en-US", voice="af_heart"))
    assert sorted(set(m["sentence_index"] for _, m in chunks_env1)) == [0, 1, 2]

    # Test _iter_audio with clause_split=True vs False
    req = backend._request(text_en, 0, None, None, {"language": "en-US", "voice": "af_heart"})
    iter_c_metas = [m for _, m in backend._iter_audio(text_en, req, sentence_mode=True, clause_split=True)]
    assert sorted(set(m["sentence_index"] for m in iter_c_metas)) == [0, 1, 2]

    iter_s_metas = [m for _, m in backend._iter_audio(text_en, req, sentence_mode=True, clause_split=False)]
    assert sorted(set(m["sentence_index"] for m in iter_s_metas)) == [0, 1]

    backend.close()


# =========================================================================
# Scheme A: Concurrency-Safe Dynamic Speaker Warm-up & Cache Tests
# =========================================================================

def _create_mock_frontend_dir(root: Path, voice_fill: dict[str, float] | None = None) -> Path:
    frontend_dir = root / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "tokens.json").write_text(json.dumps({" ": 0, "a": 1}))
    fill_map = voice_fill or {}
    for r, v in VOICES.items():
        val = fill_map.get(r, 0.0)
        data = np.full(VOICE_SHAPE, val, dtype="<f4").tobytes()
        (frontend_dir / f"{v}.bin").write_bytes(data)
    for name in ("frontend_stem.onnx", "duration_head.onnx", "target_tail.onnx"):
        (frontend_dir / name).write_bytes(b"mock-onnx")
    return frontend_dir


class _MockStemSession:
    def __init__(self, delay: float = 0.0, on_run=None):
        self.delay = delay
        self.on_run = on_run
        self.call_count = 0
        self._lock = threading.Lock()

    def run(self, out_names, feeds):
        with self._lock:
            self.call_count += 1
            if self.on_run:
                self.on_run()
        if self.delay > 0:
            import time
            time.sleep(self.delay)
        tok = feeds["input_ids"]
        n = tok.shape[1]
        return [np.zeros((1, n, 640), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)]


class _MockDurationSession:
    def run(self, out_names, feeds):
        for val in feeds.values():
            if hasattr(val, "ndim") and val.ndim == 3:
                n = val.shape[1]
                return [np.ones((1, n), dtype=np.int64) * 2]
        return [np.ones((1, 3), dtype=np.int64) * 2]


class _MockTargetTailSession:
    def run(self, out_names, feeds):
        target_durations = feeds["target_durations"]
        target_D = int(target_durations.sum())
        t = 2 * target_D
        return [
            np.zeros((1, 512, t), dtype=np.float32),
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((1, 22, 60 * t + 1), dtype=np.float32),
        ]


def _make_mock_frontend(frontend_dir: Path, stem=None, dur=None, target=None, g2p_fn=None):
    stem = stem or _MockStemSession()
    dur = dur or _MockDurationSession()
    target = target or _MockTargetTailSession()
    g2p = (lambda route: (lambda text: "a")) if g2p_fn is None else g2p_fn
    return KokoroLong32Frontend(
        frontend_dir,
        stem_session=stem,
        duration_session=dur,
        target_tail_session=target,
        g2p_factory=g2p,
        verify_manifest=False,
    )


def test_scheme_a_warmup_voice_contract_and_caching(tmp_path):
    """Verify warmup_voice returns expected timing dict and caches result."""
    fe_dir = _create_mock_frontend_dir(tmp_path)
    stem = _MockStemSession()
    fe = _make_mock_frontend(fe_dir, stem=stem)

    # Initial warmup for af_heart
    timings = fe.warmup_voice("af_heart")
    assert isinstance(timings, dict)
    assert set(timings.keys()) == {"voice_load_ms", "dry_run_ms", "total_ms"}
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())
    assert timings["total_ms"] >= timings["dry_run_ms"]
    assert fe._voice_states["a"] == VoiceWarmupState.READY
    assert "a" in fe._voice_cache
    assert stem.call_count == 1

    # Second call for the same voice (using route 'a') returns cached timings without re-running dry-run
    timings_cached = fe.warmup_voice("a")
    assert timings_cached == timings
    assert stem.call_count == 1

    # Third call using language alias 'en' also returns cached timings
    timings_en = fe.warmup_voice("en")
    assert timings_en == timings
    assert stem.call_count == 1

    # Warmup another voice: zf_xiaobei (route 'z')
    timings_z = fe.warmup_voice("zf_xiaobei")
    assert isinstance(timings_z, dict)
    assert fe._voice_states["z"] == VoiceWarmupState.READY
    assert "z" in fe._voice_cache
    assert stem.call_count == 2


def test_scheme_a_concurrent_warmup_single_flight(tmp_path):
    """Verify concurrent warmup from 8 threads triggers exactly ONE dry-run."""
    import concurrent.futures
    fe_dir = _create_mock_frontend_dir(tmp_path)
    stem = _MockStemSession(delay=0.03)  # Small delay to ensure concurrency overlap
    fe = _make_mock_frontend(fe_dir, stem=stem)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fe.warmup_voice, "af_heart") for _ in range(8)]
        results = [f.result(timeout=5.0) for f in futures]

    assert len(results) == 8
    # Stem run must have been called exactly once (single flight warmup)
    assert stem.call_count == 1
    assert fe._voice_states["a"] == VoiceWarmupState.READY
    for r in results:
        assert set(r.keys()) == {"voice_load_ms", "dry_run_ms", "total_ms"}
        assert all(isinstance(v, float) and v >= 0.0 for v in r.values())


def test_scheme_a_dynamic_speaker_switching_isolation(tmp_path):
    """Verify alternating requests between voices works seamlessly with style immutability."""
    import concurrent.futures
    # Voice 'a' filled with 1.0, voice 'z' filled with 2.0
    fe_dir = _create_mock_frontend_dir(tmp_path, voice_fill={"a": 1.0, "z": 2.0})
    fe = _make_mock_frontend(fe_dir)

    # Pre-warm both voices
    fe.warmup_voice("af_heart")
    fe.warmup_voice("zf_xiaobei")

    # Sequential alternating requests
    p_a1 = fe.prepare("a", language="en-US")
    p_z1 = fe.prepare("a", language="zh")
    p_a2 = fe.prepare("a", language="en-US")
    p_z2 = fe.prepare("a", language="zh")

    assert p_a1.route == "a" and p_a1.language == "en-US"
    assert np.all(p_a1.style == 1.0)
    assert p_z1.route == "z" and p_z1.language == "zh"
    assert np.all(p_z1.style == 2.0)
    assert p_a2.route == "a" and np.all(p_a2.style == 1.0)
    assert p_z2.route == "z" and np.all(p_z2.style == 2.0)

    # Verify style array immutability (write=False)
    assert p_a1.style.flags.writeable is False
    assert p_z1.style.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        p_a1.style[0, 0] = 999.0

    # Concurrent 16-thread stress test alternating voices
    def worker(i: int):
        if i % 2 == 0:
            prep = fe.prepare("a", language="en-US")
            assert prep.route == "a"
            assert np.all(prep.style == 1.0)
            assert prep.style.flags.writeable is False
        else:
            prep = fe.prepare("a", language="zh")
            assert prep.route == "z"
            assert np.all(prep.style == 2.0)
            assert prep.style.flags.writeable is False
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(worker, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            assert f.result() is True


def test_scheme_a_invalid_voice_names(tmp_path):
    """Verify invalid voice specifications raise ValueError."""
    fe_dir = _create_mock_frontend_dir(tmp_path)
    fe = _make_mock_frontend(fe_dir)

    for invalid in ("invalid_speaker", "", "   ", None, 12345, "unknown_lang_xyz"):
        with pytest.raises(ValueError):
            fe.warmup_voice(invalid)

    # Invalid timeout
    for bad_timeout in (-1, 0, -0.5, "five", False, True, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="timeout must be a finite positive number"):
            fe.warmup_voice("af_heart", timeout=bad_timeout)


def test_scheme_a_backend_warmup_interface(tmp_path, monkeypatch, caplog):
    """Verify KokoroDualBucketBackend warmup() method delegates correctly and enforces safety."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroDualBucketBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    # Stub frontend warmup_voice
    warmed = []
    def mock_warmup_voice(route_or_voice, timeout=5.0):
        route = KokoroLong32Frontend.resolve_route(route_or_voice)
        warmed.append((route_or_voice, route, timeout))
        return {"voice_load_ms": 1.0, "dry_run_ms": 2.0, "total_ms": 3.0}

    backend.frontend.warmup_voice = mock_warmup_voice

    # Default voice warmup (voice=None) -> warms af_heart
    t_def = backend.warmup()
    assert set(t_def.keys()) == {"voice_load_ms", "dry_run_ms", "total_ms"}
    assert warmed[-1] == ("af_heart", "a", 5.0)

    # Explicit voice warmup with custom timeout
    t_zh = backend.warmup("zf_xiaobei", timeout=10.0)
    assert set(t_zh.keys()) == {"voice_load_ms", "dry_run_ms", "total_ms"}
    assert warmed[-1] == ("zf_xiaobei", "z", 10.0)

    # Invalid voice raises ValueError
    def real_or_strict_warmup_voice(route_or_voice, timeout=5.0):
        return backend.frontend.__class__.warmup_voice(backend.frontend, route_or_voice, timeout=timeout)
    backend.frontend.warmup_voice = real_or_strict_warmup_voice

    with pytest.raises(ValueError):
        backend.warmup("unknown_speaker_xyz")

    # Invalid timeout via backend.warmup
    for bad_t in (float("nan"), float("inf"), float("-inf"), -1.0, 0):
        with pytest.raises(ValueError, match="timeout must be a finite positive number"):
            backend.warmup("af_heart", timeout=bad_t)

    # In-flight warmup tracking and warning on close()
    release_event = threading.Event()
    inflight_observed = []
    def slow_warmup(route_or_voice, timeout=5.0):
        inflight_observed.append(backend._inflight_warmups)
        release_event.wait(timeout=2.0)
        return {"voice_load_ms": 1.0, "dry_run_ms": 2.0, "total_ms": 3.0}

    backend.frontend.warmup_voice = slow_warmup
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f = ex.submit(backend.warmup, "af_heart")
        while not inflight_observed:
            time.sleep(0.01)
        assert inflight_observed[0] == 1
        assert backend._inflight_warmups == 1

        close_fut = ex.submit(backend.close)
        while not backend._closed:
            time.sleep(0.01)
        assert close_fut.running() is True
        assert backend.frontend is not None

        release_event.set()
        res = f.result(timeout=2.0)
        assert res["total_ms"] == 3.0
        close_fut.result(timeout=2.0)
        assert backend._inflight_warmups == 0
        assert backend.frontend is None

    # Calling warmup on closed backend (after preload and close) raises RuntimeError
    with pytest.raises(RuntimeError, match=f"{backend.name} is closed"):
        backend.warmup("af_heart")

    # Calling warmup on closed backend (before preload and close) raises RuntimeError
    cfg_unloaded = _create_mock_dualbucket_bundle(tmp_path / "closed_unloaded")
    backend_unloaded = DB.KokoroDualBucketBackend(cfg_unloaded, runtime_cls=MockGeneratorSession)
    backend_unloaded.close()
    with pytest.raises(RuntimeError, match=f"{backend_unloaded.name} is closed"):
        backend_unloaded.warmup("af_heart")


def test_scheme_a_warmup_failure_and_retry(tmp_path):
    """Verify failure broadcasts exception to waiters, and subsequent calls can retry."""
    import concurrent.futures
    fe_dir = _create_mock_frontend_dir(tmp_path)
    fail_event = threading.Event()
    should_fail = True

    class FailingStemSession(_MockStemSession):
        def run(self, out_names, feeds):
            if should_fail:
                fail_event.set()
                import time
                time.sleep(0.02)
                raise RuntimeError("simulated ORT dry-run failure")
            return super().run(out_names, feeds)

    stem = FailingStemSession()
    fe = _make_mock_frontend(fe_dir, stem=stem)

    # 3 concurrent threads during failure
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fe.warmup_voice, "af_heart") for _ in range(3)]
        for f in futures:
            with pytest.raises(RuntimeError, match="simulated ORT dry-run failure"):
                f.result(timeout=5.0)

    assert fe._voice_states["a"] == VoiceWarmupState.FAILED

    # Subsequent retry after fixing the issue
    should_fail = False
    timings = fe.warmup_voice("af_heart")
    assert isinstance(timings, dict)
    assert fe._voice_states["a"] == VoiceWarmupState.READY


def test_scheme_a_initiating_warmup_timeout(tmp_path):
    """Verify that if the initiating thread exceeds timeout, TimeoutError is raised,
    state transitions to FAILED, and concurrent waiters receive TimeoutError."""
    import concurrent.futures
    fe_dir = _create_mock_frontend_dir(tmp_path)

    # Stem session with delay exceeding warmup timeout
    stem = _MockStemSession(delay=0.08)
    fe = _make_mock_frontend(fe_dir, stem=stem)

    # 1. Initiating thread timeout
    with pytest.raises(TimeoutError, match="voice warmup timed out after 0.02s: af_heart"):
        fe.warmup_voice("af_heart", timeout=0.02)

    assert fe._voice_states["a"] == VoiceWarmupState.FAILED

    # 2. Concurrent waiters also get TimeoutError when initiating thread times out
    fe2 = _make_mock_frontend(fe_dir, stem=stem)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fe2.warmup_voice, "af_heart", timeout=0.02) for _ in range(3)]
        for f in futures:
            with pytest.raises(TimeoutError, match="timed out"):
                f.result(timeout=5.0)

    assert fe2._voice_states["a"] == VoiceWarmupState.FAILED

    # 3. Subsequent retry with adequate timeout succeeds
    stem.delay = 0.0
    timings = fe2.warmup_voice("af_heart", timeout=5.0)
    assert isinstance(timings, dict)
    assert fe2._voice_states["a"] == VoiceWarmupState.READY


def test_dualbucket_close_waits_for_inflight_warmup(tmp_path, monkeypatch):
    """Verify close() cleanly waits for in-flight warmup to complete before releasing resources."""
    import concurrent.futures
    import threading
    import time

    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_dualbucket_bundle(tmp_path)
    backend = DB.KokoroDualBucketBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    warmup_started = threading.Event()
    warmup_release = threading.Event()
    warmup_finished = threading.Event()

    def slow_warmup(route_or_voice, timeout=5.0):
        warmup_started.set()
        warmup_release.wait(timeout=2.0)
        warmup_finished.set()
        return {"voice_load_ms": 1.5, "dry_run_ms": 2.5, "total_ms": 4.0}

    backend.frontend.warmup_voice = slow_warmup

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        warmup_fut = ex.submit(backend.warmup, "af_heart")
        assert warmup_started.wait(timeout=2.0) is True
        assert backend._inflight_warmups == 1

        close_fut = ex.submit(backend.close, timeout=5.0)

        # Give close a moment to enter wait_for; backend._closed should be True
        while not backend._closed:
            time.sleep(0.01)

        # close() is still blocked waiting for warmup to finish
        assert close_fut.running() is True
        assert warmup_finished.is_set() is False
        # Resources must not be released while warmup is in flight
        assert backend.frontend is not None
        assert backend.generator is not None

        # Allow warmup to complete
        warmup_release.set()

        # Both should complete cleanly
        res = warmup_fut.result(timeout=2.0)
        assert res["total_ms"] == 4.0
        assert warmup_finished.is_set() is True

        close_fut.result(timeout=2.0)
        assert backend._inflight_warmups == 0
        assert backend.frontend is None
        assert backend.generator is None
        assert backend.is_ready() is False

    # Also verify timeout: if a warmup hangs, close(timeout=...) times out and releases resources
    backend2 = DB.KokoroDualBucketBackend(_create_mock_dualbucket_bundle(tmp_path / "timeout"), runtime_cls=MockGeneratorSession)
    backend2.preload()

    hang_started = threading.Event()
    hang_blocker = threading.Event()
    def hanging_warmup(route_or_voice, timeout=5.0):
        hang_started.set()
        hang_blocker.wait(timeout=5.0)
        return {}

    backend2.frontend.warmup_voice = hanging_warmup
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        hang_fut = ex.submit(backend2.warmup, "af_heart")
        assert hang_started.wait(timeout=2.0) is True
        # close with a short timeout should not hang indefinitely
        t0 = time.monotonic()
        backend2.close(timeout=0.05)
        dt = time.monotonic() - t0
        assert dt < 1.0
        assert backend2._closed is True
        assert backend2.frontend is None
        # Clean up hanging worker
        hang_blocker.set()
        hang_fut.result(timeout=2.0)


# =========================================================================
# English Streaming Clause Optimization & Word Chunking Tests
# =========================================================================

def test_chunk_by_words_basic_and_invariants(monkeypatch):
    """Test chunk_by_words helper: empty, single word, max_words, invariants."""
    # 1. Empty string
    assert DB.chunk_by_words("") == []
    assert "".join(DB.chunk_by_words("")) == ""

    # 2. Spaces only
    assert DB.chunk_by_words("   ") == ["   "]
    assert "".join(DB.chunk_by_words("   ")) == "   "

    # 3. Text without spaces (English single word, Chinese text)
    assert DB.chunk_by_words("Supercalifragilistic") == ["Supercalifragilistic"]
    zh_text = "今天天气真好，阳光明媚。"
    assert DB.chunk_by_words(zh_text) == [zh_text]

    # 4. Words <= max_words
    assert DB.chunk_by_words("hello world", 6) == ["hello world"]
    assert DB.chunk_by_words("1 2 3 4 5 6", 6) == ["1 2 3 4 5 6"]

    # 5. Words > max_words
    c7 = DB.chunk_by_words("1 2 3 4 5 6 7", 6)
    assert c7 == ["1 2 3 4 5 6 ", "7"]
    assert "".join(c7) == "1 2 3 4 5 6 7"

    # 6. Long sentence (13 words)
    long_text = "The quick brown fox jumps over the lazy dog and runs away into the forest"
    c_long = DB.chunk_by_words(long_text, 6)
    assert len(c_long) == 3
    assert c_long[0] == "The quick brown fox jumps over "
    assert c_long[1] == "the lazy dog and runs away "
    assert c_long[2] == "into the forest"
    assert "".join(c_long) == long_text

    # 7. Custom max_words
    c_custom = DB.chunk_by_words("one two three four five six seven", max_words=3)
    assert c_custom == ["one two three ", "four five six ", "seven"]
    assert "".join(c_custom) == "one two three four five six seven"

    # 8. Environment variable KOKORO_STREAM_MAX_WORDS
    monkeypatch.setenv("KOKORO_STREAM_MAX_WORDS", "4")
    c_env = DB.chunk_by_words("one two three four five six seven")
    assert c_env == ["one two three four ", "five six seven"]
    assert "".join(c_env) == "one two three four five six seven"

    # 9. Invalid env variable falls back to default 6
    monkeypatch.setenv("KOKORO_STREAM_MAX_WORDS", "invalid_value")
    c_inv = DB.chunk_by_words("1 2 3 4 5 6 7")
    assert c_inv == ["1 2 3 4 5 6 ", "7"]


def test_synthesize_stream_english_clause_word_chunking(tmp_path, monkeypatch):
    """Verify synthesize_stream splits long English clauses into sub-clauses when clause_split=True."""
    monkeypatch.setattr(KokoroLong32Frontend, "preload_sessions", lambda self: {})
    cfg = _create_mock_t50_bundle(tmp_path)
    backend = DB.KokoroDualBucketBackend(cfg, runtime_cls=MockGeneratorSession)
    backend.preload()

    def fake_prepare(text, language="en-US", speed=1.0, cancel_check=None):
        route = ROUTES.get(language, "a")
        return SimpleNamespace(
            route=route,
            language=language,
            text=text,
            input_ids=np.zeros((1, 5), dtype=np.int64),
            natural_durations=np.full(5, 4, dtype=np.int64),
            natural_D=20,
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
            "route": prepared.route,
            "language": prepared.language,
            "frontend_detail": {},
        }

    backend.frontend.prepare = fake_prepare
    backend.frontend.render = fake_render

    long_en = "The quick brown fox jumps over the lazy dog and runs away into the forest"

    # With clause_split=True (default), 13 words without punctuation get split into 3 sub-clauses
    chunks_c = list(backend.synthesize_stream(long_en, language="en-US", voice="af_heart", clause_split=True))
    assert len(chunks_c) > 0

    indices_c = sorted(set(m["sentence_index"] for _, m in chunks_c))
    assert indices_c == [0, 1, 2]

    sub_clauses = {}
    for _, meta in chunks_c:
        sub_clauses[meta["sentence_index"]] = meta["sentence_text"]

    assert sub_clauses[0] == "The quick brown fox jumps over "
    assert sub_clauses[1] == "the lazy dog and runs away "
    assert sub_clauses[2] == "into the forest"
    assert "".join(sub_clauses[i] for i in sorted(sub_clauses)) == long_en

    # With clause_split=False, no clause splitting: 1 sentence index
    chunks_s = list(backend.synthesize_stream(long_en, language="en-US", voice="af_heart", clause_split=False))
    assert len(chunks_s) > 0
    indices_s = sorted(set(m["sentence_index"] for _, m in chunks_s))
    assert indices_s == [0]
    for _, meta in chunks_s:
        assert meta["sentence_text"] == long_en

    backend.close()
