"""rkvoice-stream: Streaming speech AI on Rockchip NPU platforms."""

__version__ = "0.1.0"

from .engine.asr import create_asr
from .engine.tts import create_tts

__all__ = ["create_asr", "create_tts", "__version__"]


def load_config(path: str) -> dict:
    """Load a YAML configuration profile."""
    import yaml
    from pathlib import Path

    with open(Path(path)) as f:
        return yaml.safe_load(f)


def create_from_config(config: dict, engine: str = None):
    """Create ASR and/or TTS engines from a config dict.

    Args:
        config: Parsed YAML config (from load_config)
        engine: "asr", "tts", or None (both)

    Returns:
        Single engine if engine specified, else (asr, tts) tuple.
        Returns None for engines not configured.
    """
    asr_cfg = config.get("asr")
    tts_cfg = config.get("tts")

    asr = None
    tts = None

    if engine in (None, "asr") and asr_cfg:
        _apply_asr_env(asr_cfg)
        asr = create_asr(asr_cfg.get("backend"))

    if engine in (None, "tts") and tts_cfg:
        _apply_tts_env(tts_cfg)
        tts = create_tts(tts_cfg.get("backend"))

    if engine == "asr":
        return asr
    elif engine == "tts":
        return tts
    else:
        return asr, tts


def _apply_asr_env(cfg: dict) -> None:
    """Apply config keys understood by current ASR backends."""
    import os
    from pathlib import Path

    backend = cfg.get("backend")
    model_dir = cfg.get("model_dir")
    if backend:
        os.environ["ASR_BACKEND"] = str(backend)
    if model_dir:
        if backend == "paraformer_rknn" or backend == "paraformer_sherpa":
            os.environ["PARAFORMER_MODEL_DIR"] = str(model_dir)
            os.environ["PARAFORMER_RKNN_DIR"] = str(Path(model_dir) / "rknn")
        elif backend == "sensevoice_sherpa":
            os.environ["SENSEVOICE_MODEL_DIR"] = str(model_dir)
        else:
            os.environ["ASR_MODEL_DIR"] = str(model_dir)
    if "rknn_dir" in cfg:
        os.environ["PARAFORMER_RKNN_DIR"] = str(cfg["rknn_dir"])
    if "precision" in cfg:
        os.environ["PARAFORMER_RKNN_PRECISION"] = str(cfg["precision"])
    if "encoder_precision" in cfg:
        os.environ["PARAFORMER_RKNN_ENC_PRECISION"] = str(cfg["encoder_precision"])
    if "decoder_precision" in cfg:
        os.environ["PARAFORMER_RKNN_DEC_PRECISION"] = str(cfg["decoder_precision"])
    if "encoder_mode" in cfg:
        os.environ["PARAFORMER_RKNN_ENCODER_MODE"] = str(cfg["encoder_mode"])
    if "decoder_backend" in cfg:
        os.environ["PARAFORMER_RKNN_DECODER"] = str(cfg["decoder_backend"])
    if "encoder_suffix_onnx" in cfg:
        os.environ["PARAFORMER_ENCODER_SUFFIX_ONNX"] = str(cfg["encoder_suffix_onnx"])
    if "decoder_onnx" in cfg:
        os.environ["PARAFORMER_DECODER_ONNX"] = str(cfg["decoder_onnx"])
    if "encoder_core" in cfg:
        os.environ["PARAFORMER_RKNN_ENC_CORE"] = str(cfg["encoder_core"])
    if "decoder_core" in cfg:
        os.environ["PARAFORMER_RKNN_DEC_CORE"] = str(cfg["decoder_core"])


def _apply_tts_env(cfg: dict) -> None:
    """Apply config keys understood by current TTS backends."""
    import os

    backend = cfg.get("backend")
    model_dir = cfg.get("model_dir")
    if backend:
        os.environ["TTS_BACKEND"] = str(backend)
    if model_dir:
        if backend == "piper_rknn":
            os.environ["PIPER_MODEL_DIR"] = str(model_dir)
        elif backend in ("kokoro_rknn", "kokora_rknn"):
            os.environ["KOKORO_MODEL_DIR"] = str(model_dir)
            os.environ["TTS_MODEL_DIR"] = str(model_dir)
            os.environ["MODEL_DIR"] = str(model_dir)
        else:
            os.environ["TTS_MODEL_DIR"] = str(model_dir)
            os.environ["MODEL_DIR"] = str(model_dir)
    if backend in ("kokoro_rknn", "kokora_rknn"):
        mapping = {
            "mode": "KOKORO_RKNN_MODE",
            "model": "KOKORO_MODEL",
            "prefix_onnx": "KOKORO_PREFIX_ONNX",
            "front_rknn": "KOKORO_FRONT_RKNN",
            "tail_onnx": "KOKORO_TAIL_ONNX",
            "seq_len": "KOKORO_SEQ_LEN",
            "sample_rate": "KOKORO_SAMPLE_RATE",
            "voice": "KOKORO_VOICE",
            "prefix_ort_intra_op": "KOKORO_PREFIX_ORT_INTRA_OP",
            "prefix_ort_inter_op": "KOKORO_PREFIX_ORT_INTER_OP",
            "tail_ort_intra_op": "KOKORO_TAIL_ORT_INTRA_OP",
            "tail_ort_inter_op": "KOKORO_TAIL_ORT_INTER_OP",
            "ort_graph_opt": "KOKORO_ORT_GRAPH_OPT",
            "ort_enable_cpu_mem_arena": "KOKORO_ORT_ENABLE_CPU_MEM_ARENA",
            "ort_enable_mem_pattern": "KOKORO_ORT_ENABLE_MEM_PATTERN",
            "ort_enable_mem_reuse": "KOKORO_ORT_ENABLE_MEM_REUSE",
        }
        for key, env_name in mapping.items():
            if key in cfg:
                os.environ[env_name] = str(cfg[key])
