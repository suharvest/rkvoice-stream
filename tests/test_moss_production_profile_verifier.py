"""Unit checks for the MOSS production profile verifier."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_production_profile.py"
    spec = importlib.util.spec_from_file_location("verify_moss_production_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_production_profile_summary_requires_all_gates():
    verifier = _load_verifier()

    passed = verifier._summarize(
        artifact_manifest={"artifacts": []},
        service_report={"gates": {"passed": True}},
        backend_stage_report={"gates": {"passed": True}},
        roundtrip_report={"gates": {"passed": True}},
        errors=[],
    )
    failed = verifier._summarize(
        artifact_manifest={"artifacts": []},
        service_report={"gates": {"passed": True}},
        backend_stage_report={"gates": {"passed": True}},
        roundtrip_report={"gates": {"passed": False}},
        errors=[],
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["roundtrip_quality"] is False


def test_production_profile_builds_manifest_locked_commands(tmp_path):
    verifier = _load_verifier()
    args = argparse.Namespace(
        repo_root=tmp_path / "repo",
        model_dir=tmp_path / "moss",
        manifest="moss-ort-manifest.json",
        port=8624,
        text="你好",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        max_tts_wall_ms=2000.0,
        max_dialogue_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        max_dialogue_first_payload_ms=1500.0,
        max_dialogue_payload_gap_ms=1500.0,
        startup_timeout=180.0,
        request_timeout=120.0,
        log_file=tmp_path / "uvicorn.log",
        asr_model_dir=tmp_path / "asr",
        roundtrip_out_dir=tmp_path / "roundtrip",
        threads=6,
        prefill_threads=8,
        decode_threads=None,
        sampler_threads=None,
        codec_threads=5,
        codec_batch_frames=3,
        min_dialogue_binary_chunks=7,
        roundtrip_max_new_frames=20,
        voice="Junhao",
        seed=314,
        prefill_seq=0,
        warmup_text="你好",
        max_avg_cer=0.5,
        max_cer=1.0,
        min_rms=0.02,
        max_roundtrip_ttfa_ms=1500.0,
        min_backend_audio_frames=20,
        max_backend_ttfa_ms=1500.0,
        max_backend_prefill_ms=1200.0,
        max_codec_ms=120.0,
        sentences=None,
        service_max_new_frames=20,
    )

    service_cmd = verifier._build_service_cmd(args, tmp_path / "service.json")
    backend_stage_cmd = verifier._build_backend_stage_cmd(args, tmp_path / "backend_stage.json")
    roundtrip_cmd = verifier._build_roundtrip_cmd(args, tmp_path / "roundtrip.json")

    assert "--set-env" in service_cmd
    assert f"MOSS_ORT_MANIFEST={args.manifest}" in service_cmd
    assert "MOSS_ORT_PREFILL_SEQ=0" in service_cmd
    assert "MOSS_ORT_MAX_NEW_FRAMES=20" in service_cmd
    assert "MOSS_ORT_LOAD_FULL_CODEC=0" in service_cmd
    assert "MOSS_ORT_PREFILL_THREADS=8" in service_cmd
    assert "MOSS_ORT_CODEC_THREADS=5" in service_cmd
    assert "MOSS_ORT_CODEC_BATCH_FRAMES=3" in service_cmd
    assert "MOSS_ORT_CODEC_ASYNC=0" in service_cmd
    assert "MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK=0" in service_cmd
    assert "MOSS_ORT_HYBRID_RKNN=0" in service_cmd
    assert "--require-manifest-validated" in service_cmd
    assert service_cmd[service_cmd.index("--expected-voice") + 1] == "Junhao"
    assert service_cmd[service_cmd.index("--expected-seed") + 1] == "314"
    assert service_cmd[service_cmd.index("--expected-manifest") + 1] == "moss-ort-manifest.json"
    assert service_cmd[service_cmd.index("--expected-codec-batch-frames") + 1] == "3"
    assert "--require-production-runtime" in service_cmd
    assert service_cmd[service_cmd.index("--max-dialogue-payload-gap-ms") + 1] == "1500.0"
    assert service_cmd[service_cmd.index("--min-dialogue-binary-chunks") + 1] == "7"
    assert backend_stage_cmd[backend_stage_cmd.index("--max-new-frames") + 1] == "20"
    assert backend_stage_cmd[backend_stage_cmd.index("--min-audio-frames") + 1] == "20"
    assert backend_stage_cmd[backend_stage_cmd.index("--max-ttfa-ms") + 1] == "1500.0"
    assert backend_stage_cmd[backend_stage_cmd.index("--max-prefill-ms") + 1] == "1200.0"
    assert backend_stage_cmd[backend_stage_cmd.index("--max-codec-ms") + 1] == "120.0"
    assert "--manifest" in roundtrip_cmd
    assert roundtrip_cmd[roundtrip_cmd.index("--manifest") + 1] == args.manifest
    assert "--seed" in roundtrip_cmd
    assert roundtrip_cmd[roundtrip_cmd.index("--seed") + 1] == "314"
    assert roundtrip_cmd[roundtrip_cmd.index("--prefill-threads") + 1] == "8"
    assert roundtrip_cmd[roundtrip_cmd.index("--decode-threads") + 1] == "6"
    assert roundtrip_cmd[roundtrip_cmd.index("--sampler-threads") + 1] == "6"
    assert roundtrip_cmd[roundtrip_cmd.index("--codec-threads") + 1] == "5"
    assert roundtrip_cmd[roundtrip_cmd.index("--codec-batch-frames") + 1] == "3"


def test_production_profile_cli_defaults_match_current_contract(tmp_path):
    verifier = _load_verifier()
    parser = verifier._build_parser()

    args = parser.parse_args(
        [
            "--model-dir",
            str(tmp_path / "moss"),
            "--asr-model-dir",
            str(tmp_path / "asr"),
        ]
    )

    assert verifier.PRODUCTION_CODEC_BATCH_FRAMES == 3
    assert args.codec_batch_frames == 3
    assert args.warmup_text == "你好"
