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
    platform = config.get("platform", "rk3576")
    asr_cfg = config.get("asr")
    tts_cfg = config.get("tts")

    asr = None
    tts = None

    if engine in (None, "asr") and asr_cfg:
        asr = create_asr(platform=platform, **asr_cfg)

    if engine in (None, "tts") and tts_cfg:
        tts = create_tts(platform=platform, **tts_cfg)

    if engine == "asr":
        return asr
    elif engine == "tts":
        return tts
    else:
        return asr, tts
