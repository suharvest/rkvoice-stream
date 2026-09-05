"""Host-only factory/config wiring; no NPU or bundle qualification claims."""
from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

import rkvoice_stream
from rkvoice_stream.engine.tts import create_backend


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def _stub(monkeypatch, module_name, class_name):
    module = ModuleType(module_name)
    cls = type(class_name, (), {})
    setattr(module, class_name, cls)
    monkeypatch.setitem(sys.modules, module_name, module)
    return cls


@pytest.mark.parametrize("explicit", [True, False])
def test_convonly_factory_is_distinct(monkeypatch, explicit):
    cls = _stub(monkeypatch, "rkvoice_stream.backends.tts.kokoro_convonly", "KokoroConvOnlyBackend")
    monkeypatch.setenv("TTS_BACKEND", "kokoro_convonly")
    assert isinstance(create_backend("kokoro_convonly" if explicit else None), cls)


@pytest.mark.parametrize("name", ["kokoro_rknn", "kokora_rknn"])
def test_legacy_kokoro_aliases_are_unchanged(monkeypatch, name):
    cls = _stub(monkeypatch, "rkvoice_stream.backends.tts.kokoro_rknn", "KokoroRKNNBackend")
    assert isinstance(create_backend(name), cls)


def test_factory_default_remains_qwen3(monkeypatch):
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    cls = _stub(monkeypatch, "rkvoice_stream.backends.tts.qwen3_rknn", "Qwen3RKNNBackend")
    assert isinstance(create_backend(), cls)


def test_unknown_backend_does_not_fall_back():
    with pytest.raises(ValueError, match="Unknown TTS backend"):
        create_backend("kokoro_convonly_typo")


def test_yaml_maps_only_the_explicit_convonly_contract(monkeypatch):
    for key in ("MODEL_DIR", "TTS_MODEL_DIR", "KOKORO_MODEL_DIR", "KOKORO_RKNN_MODE"):
        monkeypatch.setenv(key, "legacy-sentinel")
    rkvoice_stream._apply_tts_env({
        "backend": "kokoro_convonly", "bundle_root": "/qualified/rk3576",
        "manifest_sha256": "", "platform": "rk3576", "intra": 6, "inter": 1,
        "model_dir": "/must-not-be-used", "mode": "hybrid", "require_backend": 1,
    })
    assert os.environ["TTS_BACKEND"] == "kokoro_convonly"
    assert os.environ["REQUIRE_TTS_BACKEND"] == "1"
    assert os.environ["KOKORO_CONVONLY_ROOT"] == "/qualified/rk3576"
    assert os.environ["KOKORO_CONVONLY_MANIFEST_SHA256"] == ""
    assert os.environ["RK_PLATFORM"] == "rk3576"
    assert os.environ["KOKORO_FRONTEND_INTRA_OP_THREADS"] == "6"
    assert os.environ["KOKORO_FRONTEND_INTER_OP_THREADS"] == "1"
    for key in ("MODEL_DIR", "TTS_MODEL_DIR", "KOKORO_MODEL_DIR", "KOKORO_RKNN_MODE"):
        assert os.environ[key] == "legacy-sentinel"


def test_omitted_yaml_fields_preserve_operator_values(monkeypatch):
    monkeypatch.setenv("KOKORO_FRONTEND_INTRA_OP_THREADS", "0")
    monkeypatch.setenv("KOKORO_CONVONLY_ROOT", "/operator/bundle")
    rkvoice_stream._apply_tts_env({"backend": "kokoro_convonly", "inter": 1})
    assert os.environ["KOKORO_FRONTEND_INTRA_OP_THREADS"] == "0"
    assert os.environ["KOKORO_CONVONLY_ROOT"] == "/operator/bundle"


def test_create_from_config_applies_mapping_before_factory(monkeypatch):
    seen = {}
    sentinel = object()
    def factory(name):
        seen.update(name=name, root=os.environ["KOKORO_CONVONLY_ROOT"])
        return sentinel
    monkeypatch.setattr(rkvoice_stream, "create_tts", factory)
    result = rkvoice_stream.create_from_config({"tts": {
        "backend": "kokoro_convonly", "bundle_root": "/qualified/rk3588",
    }}, engine="tts")
    assert result is sentinel
    assert seen == {"name": "kokoro_convonly", "root": "/qualified/rk3588"}


def test_legacy_yaml_mapping_is_unchanged():
    rkvoice_stream._apply_tts_env({"backend": "kokoro_rknn", "model_dir": "/legacy", "mode": "long32"})
    assert os.environ["KOKORO_MODEL_DIR"] == "/legacy"
    assert os.environ["TTS_MODEL_DIR"] == "/legacy"
    assert os.environ["MODEL_DIR"] == "/legacy"
    assert os.environ["KOKORO_RKNN_MODE"] == "long32"
