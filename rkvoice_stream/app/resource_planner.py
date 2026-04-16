"""ResourcePlanner for automatic ASR/TTS backend selection.

Auto-selects optimal ASR/TTS backends based on SPEECH_MODE env var.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

logger = logging.getLogger(__name__)

_MODE_PLANS = {
    "dialogue": {
        "asr": {"backend": "qwen3_asr_rk", "provider": "npu"},
        "tts": {"backend": "matcha_rknn", "provider": "npu"},
        "parallel": False,
    },
    "interpret": {
        "asr": {"backend": "sensevoice_sherpa", "provider": "cpu"},
        "tts": {"backend": "matcha_rknn", "provider": "npu"},
        "parallel": True,
    },
    "asr_only": {
        "asr": {"backend": "qwen3_asr_rk", "provider": "npu"},
        "tts": None,
        "parallel": False,
    },
    "tts_only": {
        "asr": None,
        "tts": {"backend": "matcha_rknn", "provider": "npu"},
        "parallel": False,
    },
}


class ResourcePlanner:
    """Auto-select ASR/TTS backends based on SPEECH_MODE env var."""

    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or os.environ.get("SPEECH_MODE", "dialogue")

    def plan(self) -> dict:
        """
        Returns config dict:
        {
            "mode": "dialogue",
            "asr": {"backend": "qwen3_asr_rk", "provider": "npu"},
            "tts": {"backend": "matcha_rknn", "provider": "npu"},
            "parallel": False,
        }
        """
        if self.mode not in _MODE_PLANS:
            logger.warning(
                "Invalid SPEECH_MODE=%r, falling back to 'dialogue'. Valid: %s",
                self.mode, list(_MODE_PLANS.keys())
            )
            self.mode = "dialogue"

        cfg = _MODE_PLANS[self.mode]

        result = {
            "mode": self.mode,
            "asr": cfg["asr"],
            "tts": cfg["tts"],
            "parallel": cfg["parallel"],
        }

        logger.info(
            "ResourcePlanner: mode=%s asr=%s tts=%s parallel=%s",
            result["mode"],
            f"{result['asr']['backend']}({result['asr']['provider']})" if result["asr"] else "none",
            f"{result['tts']['backend']}({result['tts']['provider']})" if result["tts"] else "none",
            result["parallel"],
        )

        return result


def get_resource_plan() -> dict:
    return ResourcePlanner().plan()
