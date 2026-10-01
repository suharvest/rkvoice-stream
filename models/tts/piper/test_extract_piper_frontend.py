from pathlib import Path

import onnx
from onnx import TensorProto, helper

from extract_piper_frontend import DEFAULT_OUTPUTS, extract
import numpy as np
import pytest
import json
import types
import sys
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parents[3]))
import rkvoice_stream.backends.tts.piper as piper_backend
from rkvoice_stream.backends.tts.piper import _LangModel, PiperRKNNBackend


def _model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.INT64, [1, 4]),
        helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("x_mask", TensorProto.FLOAT, [1, 1, 4]),
    ]
    nodes = [
        helper.make_node("Cast", ["input"], ["emb"], name="/enc_p/embed", to=TensorProto.FLOAT),
        helper.make_node("Identity", ["emb"], ["/enc_p/Cast_1_output_0"], name="/enc_p/Cast_1"),
        helper.make_node("Identity", ["emb"], ["/enc_p/encoder/Mul_2_output_0"], name="/enc_p/encoder/Mul_2"),
        helper.make_node("Identity", ["emb"], ["/enc_p/Split_output_0"], name="/enc_p/Split0"),
        helper.make_node("Identity", ["emb"], ["/enc_p/Split_output_1"], name="/enc_p/Split1"),
        helper.make_node("Unsqueeze", ["x_mask", "axes"], ["/enc_p/encoder/Unsqueeze_output_0"], name="/enc_p/encoder/Unsqueeze"),
        helper.make_node("Add", ["/enc_p/Split_output_0", "/enc_p/Split_output_1"], ["z"], name="/dp/Add"),
        helper.make_node("Identity", ["/enc_p/encoder/Unsqueeze_output_0"], ["y_mask"], name="/dp/mask"),
    ]
    graph = helper.make_graph(nodes, "piper", inputs,
                              [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 192, 4]),
                               helper.make_tensor_value_info("y_mask", TensorProto.FLOAT, [1, 1, 4])])
    graph.initializer.append(helper.make_tensor("axes", TensorProto.INT64, [1], [2]))
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), path)


def test_extract_is_dependency_closed_and_keeps_mask_input(tmp_path):
    source = tmp_path / "model.onnx"
    _model(source)
    out = tmp_path / "split"
    extract(source, out, 4, tuple([
        "/enc_p/Cast_1_output_0", "/enc_p/encoder/Mul_2_output_0",
        "/enc_p/Split_output_0", "/enc_p/Split_output_1", "/enc_p/encoder/Unsqueeze_output_0",
    ]), True)
    prefix = onnx.load(out / "text_encoder.onnx")
    remainder = onnx.load(out / "remainder.onnx")
    assert "x_mask" in {i.name for i in prefix.graph.input}
    assert "/dp/Add" not in {n.name for n in prefix.graph.node}
    assert "/enc_p/encoder/Mul_2" not in {n.name for n in remainder.graph.node}
    assert [i.name for i in remainder.graph.input if i.name.startswith("/enc_p/")] == [
        "/enc_p/Split_output_0", "/enc_p/Split_output_1",
        "/enc_p/encoder/Unsqueeze_output_0",
    ]


def _conditioned_model(path: Path, random_conditioner: bool = False,
                       extra_dp: bool = False) -> None:
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.INT64, [1, 4]),
        helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("x_mask", TensorProto.FLOAT, [1, 1, 4]),
    ]
    nodes = [
        helper.make_node("Identity", ["x_mask"], ["/enc_p/Cast_1_output_0"], name="/enc_p/Cast_1"),
        helper.make_node("Identity", ["hidden_seed"], ["/enc_p/encoder/Mul_2_output_0"], name="/enc_p/encoder/Mul_2"),
        helper.make_node("Identity", ["/enc_p/encoder/Mul_2_output_0"], ["/enc_p/Split_output_0"], name="/enc_p/Split0"),
        helper.make_node("Identity", ["extra_dp" if extra_dp else "/enc_p/encoder/Mul_2_output_0"], ["/enc_p/Split_output_1"], name="/enc_p/Split1"),
        helper.make_node("Unsqueeze", ["x_mask", "axes"], ["/enc_p/encoder/Unsqueeze_output_0"], name="/enc_p/encoder/Unsqueeze"),
    ]
    if extra_dp:
        nodes.insert(3, helper.make_node("Identity", ["/enc_p/encoder/Mul_2_output_0"], ["extra_dp"], name="/dp/extra"))
    nodes.append(helper.make_node("Mul", ["/enc_p/encoder/Mul_2_output_0", "x_mask"], ["dp_masked"], name="/dp/mask/Mul"))
    conditioner_input = "dp_masked"
    if random_conditioner:
        nodes.append(helper.make_node("RandomNormalLike", [conditioner_input], ["random_cond"], name="/dp/pre/random"))
        conditioner_input = "random_cond"
    nodes.extend([
        helper.make_node("Identity", [conditioner_input], ["dp_pre"], name="/dp/pre/0"),
        helper.make_node("Identity", ["dp_pre"], ["dp_convs"], name="/dp/convs/0"),
        helper.make_node("Identity", ["dp_convs"], ["dp_proj"], name="/dp/proj/0"),
        helper.make_node("Add", ["dp_proj", "/enc_p/Split_output_0"], ["flow_z"], name="/dp/flows.0"),
        helper.make_node("Identity", ["flow_z"], ["z"], name="/dp/z"),
        helper.make_node("Squeeze", ["/enc_p/encoder/Unsqueeze_output_0", "axes"], ["y_mask"], name="/dp/mask"),
    ])
    graph = helper.make_graph(
        nodes, "piper_conditioned", inputs,
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 192, 4]),
         helper.make_tensor_value_info("y_mask", TensorProto.FLOAT, [1, 1, 4])],
        initializer=[helper.make_tensor("hidden_seed", TensorProto.FLOAT, [1, 192, 4], [0.25] * (1 * 192 * 4)),
                     helper.make_tensor("axes", TensorProto.INT64, [1], [2])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), path)


def test_dp_conditioner_opt_in_is_equivalent_and_default_is_unchanged(tmp_path):
    source = tmp_path / "conditioned.onnx"
    _conditioned_model(source)
    default = tmp_path / "default"
    extract(source, default, 4, tuple(DEFAULT_OUTPUTS), False)
    assert json.loads((default / "manifest.json").read_text())["dp_conditioner"]["enabled"] is False
    enabled = tmp_path / "enabled"
    manifest = extract(source, enabled, 4, tuple(DEFAULT_OUTPUTS), False, True)
    assert manifest["dp_conditioner"]["enabled"] is True
    conditioner = manifest["dp_conditioner"]["tensor"]["name"]
    prefix = onnx.load(enabled / "text_encoder.onnx")
    remainder = onnx.load(enabled / "remainder.onnx")
    assert conditioner in {o.name for o in prefix.graph.output}
    assert {"/dp/pre/0", "/dp/convs/0", "/dp/proj/0", "/dp/mask/Mul"}.issubset({n.name for n in prefix.graph.node})
    assert not any(n.name.startswith("/dp/pre/") or n.name.startswith("/dp/convs/") for n in remainder.graph.node)
    assert not any(n.name.startswith("/dp/flows.") for n in prefix.graph.node)

    feeds = {"input": np.zeros((1, 4), np.int64), "input_lengths": np.array([4], np.int64),
             "scales": np.ones(3, np.float32), "x_mask": np.ones((1, 1, 4), np.float32)}
    full = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)
    p_session = ort.InferenceSession(str(enabled / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    p_inputs = {i.name for i in p_session.get_inputs()}
    p_values = p_session.run(None, {name: feeds[name] for name in p_inputs})
    r_session = ort.InferenceSession(str(enabled / "remainder.onnx"), providers=["CPUExecutionProvider"])
    r_inputs = {i.name for i in r_session.get_inputs()}
    p_feeds = {name: feeds[name] for name in feeds if name in r_inputs}
    p_feeds.update({o.name: value for o, value in zip(p_session.get_outputs(), p_values)
                    if o.name in r_inputs})
    rem = r_session.run(None, p_feeds)
    np.testing.assert_allclose(full[0], rem[0], rtol=0, atol=0)
    np.testing.assert_allclose(full[1], rem[1], rtol=0, atol=0)


def test_dp_conditioner_opt_in_rejects_random_dependency(tmp_path):
    source = tmp_path / "unsafe.onnx"
    _conditioned_model(source, random_conditioner=True)
    with pytest.raises((RuntimeError, ValueError), match="unsafe"):
        extract(source, tmp_path / "unsafe_out", 4, tuple(DEFAULT_OUTPUTS), False, True)


def test_dp_conditioner_opt_in_rejects_dp_node_outside_safe_closure(tmp_path):
    source = tmp_path / "extra.onnx"
    _conditioned_model(source, extra_dp=True)
    output = tmp_path / "extra_out"
    with pytest.raises(RuntimeError, match="non-conditioner"):
        extract(source, output, 4, tuple(DEFAULT_OUTPUTS), False, True)
    assert not (output / "text_encoder.onnx").exists()
    assert not (output / "remainder.onnx").exists()


class _FakeRKNN:
    def inference(self, inputs):
        if len(inputs) == 4:
            return [np.zeros((1, 1, 2), np.float32), np.zeros((1, 192, 2), np.float32),
                    np.zeros((1, 192, 2), np.float32), np.zeros((1, 192, 2), np.float32)]
        return [np.ones((1, 512), np.float32)]

    def release(self):
        pass


class _FakeRemainder:
    def __init__(self, cumulative_shape):
        self.cumulative_shape = cumulative_shape

    def get_inputs(self):
        return [type("I", (), {"name": "input", "shape": [1, 4]}),
                type("I", (), {"name": "input_lengths", "shape": [1]}),
                type("I", (), {"name": "scales", "shape": [3]}),
                type("I", (), {"name": "cumulative_durations", "shape": self.cumulative_shape, "type": "tensor(float)"}),
                type("I", (), {"name": "unknown", "shape": [1]})]

    def run(self, _, feeds):
        assert "unknown" not in feeds
        assert feeds["cumulative_durations"].dtype == np.float32
        return [np.zeros((1, 192, 2), np.float32), np.ones((1, 1, 2), np.float32)]


@pytest.mark.parametrize("cumulative_shape", ([5], [4, 1]))
def test_frontend_runtime_returns_decoder_audio_and_rejects_unknown_manifest_input(cumulative_shape):
    m = _LangModel("en_US", Path("/tmp"))
    m.seq_len = 4; m.mel_len = 2; m._frontend_npu = True
    m._frontend_rknn = _FakeRKNN(); m._rknn = _FakeRKNN(); m._remainder = _FakeRemainder(cumulative_shape)
    m._frontend_manifest = {"prefix": {"inputs": ["input", "input_lengths", "scales", "x_mask"],
        "outputs": [{"name": "mask"}, {"name": "hidden"}, {"name": "mp"}, {"name": "logs"}]}}
    audio = m._infer_frontend_npu([1, 2], 1.0, 0.0, 0.0)
    assert audio.shape == (512,)
    assert float(audio[0]) == 1.0


def test_frontend_runtime_bucket_boundary_rejects_without_slicing():
    m = _LangModel("en_US", Path("/tmp"))
    m.seq_len = 4; m.mel_len = 2; m._frontend_npu = True
    m._frontend_rknn = _FakeRKNN(); m._rknn = _FakeRKNN(); m._remainder = _FakeRemainder([5])
    m._frontend_manifest = {"prefix": {"inputs": ["input", "input_lengths", "scales", "x_mask"],
        "outputs": [{"name": "mask"}, {"name": "hidden"}, {"name": "mp"}, {"name": "logs"}]}}
    audio = m._infer_frontend_npu([1, 2, 3, 4], 1.0, 0.0, 0.0)
    assert audio.shape == (512,)
    with pytest.raises(ValueError, match=r"at most 4 tokens, got 5"):
        m._infer_frontend_npu([1, 2, 3, 4, 5], 1.0, 0.0, 0.0)


def test_frontend_high_level_bucket_boundary_rejects_without_slicing(monkeypatch):
    backend = PiperRKNNBackend()
    calls = []
    monkeypatch.setattr(piper_backend, "text_to_phonemes", lambda text, voice: "PHONEMES")
    monkeypatch.setattr(piper_backend, "phonemes_to_ids", lambda phonemes, ids: [1, 2, 3, 4, 5])
    class FakeModel:
        espeak_voice = "en-us"
        phoneme_id_map = {}
        seq_len = 4
        _frontend_npu = True
        sample_rate = 22050
        length_scale = noise_scale = noise_w = 1.0
        def infer(self, *args):
            calls.append(args)
            return np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match=r"at most 4 tokens, got 5"):
        backend._synthesize_segment("too long", FakeModel())
    assert calls == []


def test_frontend_enabled_missing_artifacts_fails_explicitly(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("PIPER_ENABLE_FRONTEND_NPU", "1")
    with pytest.raises(FileNotFoundError, match="text_encoder.rknn"):
        _LangModel("en_US", tmp_path).load()


@pytest.mark.parametrize("failure", ("frontend", "decoder", "ort", "probe"))
def test_frontend_partial_load_releases_initialized_contexts(tmp_path, monkeypatch, failure):
    class FakeRKNN:
        instances = []
        def __init__(self, **_): self.releases = 0; FakeRKNN.instances.append(self)
        def load_rknn(self, _): return 1 if failure == "frontend" and len(FakeRKNN.instances) == 1 else 0
        def init_runtime(self): return 1 if failure == "decoder" and len(FakeRKNN.instances) == 2 else 0
        def inference(self, **_): return None if failure == "probe" else [np.ones((1, 512), np.float32)]
        def release(self): self.releases += 1

    class FakeORT:
        def __init__(self, *_args, **_kwargs):
            if failure == "ort": raise RuntimeError("ort injected failure")

    fake_rknnlite = types.ModuleType("rknnlite")
    fake_api = types.ModuleType("rknnlite.api"); fake_api.RKNNLite = FakeRKNN
    fake_rknnlite.api = fake_api
    monkeypatch.setitem(sys.modules, "rknnlite", fake_rknnlite)
    monkeypatch.setitem(sys.modules, "rknnlite.api", fake_api)
    fake_ort = types.ModuleType("onnxruntime"); fake_ort.InferenceSession = FakeORT
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    manifest = {"bucket": {"input": [1, 4]}, "remainder": {
        "outputs": ["z", "mask"], "output_semantic": ["z", "y_mask"],
        "output_shapes": [[1, 192, "T"], [1, 1, "T"]]},
        "prefix": {"inputs": []}}
    mpath = tmp_path / "manifest.json"; mpath.write_text(json.dumps(manifest))
    for name in ("text_encoder.rknn", "remainder.onnx", "flow_decoder.rknn"):
        (tmp_path / name).write_bytes(b"fixture")
    model = _LangModel("en_US", tmp_path)
    with pytest.raises(RuntimeError):
        model._load_frontend_npu(tmp_path / "text_encoder.rknn", tmp_path / "remainder.onnx", mpath)
    assert model._frontend_rknn is None
    assert model._rknn is None
    assert all(x.releases == 1 for x in FakeRKNN.instances)
