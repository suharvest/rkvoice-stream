"""Unit checks for the MOSS production entrypoint verifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = ROOT / "models" / "tts" / "moss" / "verify_moss_production_entrypoint.py"
    spec = importlib.util.spec_from_file_location("verify_moss_production_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_production_entrypoint_builds_runner_command(tmp_path):
    verifier = _load_verifier()

    command = verifier.build_runner_command(
        runner=Path("models/tts/moss/run_moss_production_server.py"),
        python="/venv/bin/python",
        config=Path("configs/rk3576-moss-ort-stream.yaml"),
        host="127.0.0.1",
        port=8628,
        preflight_json=tmp_path / "preflight.json",
    )

    assert command == [
        "/venv/bin/python",
        "models/tts/moss/run_moss_production_server.py",
        "--config",
        "configs/rk3576-moss-ort-stream.yaml",
        "--host",
        "127.0.0.1",
        "--port",
        "8628",
        "--python",
        "/venv/bin/python",
        "--json-out",
        str(tmp_path / "preflight.json"),
    ]


def test_production_entrypoint_augments_existing_report(tmp_path):
    verifier = _load_verifier()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"gates": {"passed": True}}), encoding="utf-8")

    verifier._augment_report(report, entrypoint={"runner": "runner.py", "preflight_json": "preflight.json"})

    parsed = json.loads(report.read_text(encoding="utf-8"))
    assert parsed["gates"]["passed"] is True
    assert parsed["entrypoint"] == {"runner": "runner.py", "preflight_json": "preflight.json"}


def test_production_entrypoint_source_uses_string_json_out_in_forwarded_argv():
    source = (ROOT / "models" / "tts" / "moss" / "verify_moss_production_entrypoint.py").read_text(encoding="utf-8")

    assert 'sys.argv.extend(["--json-out", str(args.json_out)])' in source
