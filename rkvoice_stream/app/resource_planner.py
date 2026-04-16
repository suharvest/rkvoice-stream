"""ResourcePlanner for automatic ASR/TTS backend selection.

Auto-selects optimal ASR/TTS backends based on SPEECH_MODE env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackendPlan:
    """Backend configuration plan."""
    backend: str
    provider: str  # "npu" or "cpu"
    env: dict = field(default_factory=dict)


class ResourcePlanner:
    """Auto-select ASR/TTS backends based on SPEECH_MODE env var."""

    MODES = {
        "dialogue": {
            "description": "ASR + TTS serial (conversation). Both use NPU.",
            "parallel": False,
        },
        "interpret": {
            "description": "ASR + TTS parallel (simultaneous interpretation). ASR on CPU, TTS on NPU.",
            "parallel": True,
        },
        "asr_only": {
            "description": "Only ASR, all NPU resources to ASR.",
            "parallel": False,
        },
        "tts_only": {
            "description": "Only TTS.",
            "parallel": False,
        },
    }

    def plan(self, mode: str = None, platform: str = "rk3576") -> dict:
        """
        Returns config dict:
        {
            "mode": "dialogue",
            "asr": {
                "backend": "qwen3_asr_rk",
                "provider": "npu",
                "env": {"ASR_BACKEND": "qwen3_asr_rk", ...}
            },
            "tts": {
                "backend": "matcha_rknn",
                "provider": "npu",
                "env": {"TTS_BACKEND": "matcha_rknn", ...}
            },
            "parallel": False,  # whether ASR and TTS can run simultaneously
            "warnings": [],
        }
        """
        if mode is None:
            mode = os.environ.get("SPEECH_MODE", "dialogue")

        if mode not in self.MODES:
            raise ValueError(
                f"Invalid SPEECH_MODE: {mode!r}. "
                f"Valid modes: {list(self.MODES.keys())}"
            )

        warnings = []
        mode_config = self.MODES[mode]

        # Check platform for NPU core allocation
        if platform not in ["rk3576", "rk3588"]:
            warnings.append(f"Unknown platform: {platform!r}, using defaults")

        # Build plan based on mode
        if mode == "dialogue":
            # ASR + TTS serial, both use NPU
            # ASR: qwen3_asr_rk (best quality) + TTS: matcha_rknn
            asr_plan = BackendPlan(
                backend="qwen3_asr_rk",
                provider="npu",
                env={"ASR_BACKEND": "qwen3_asr_rk"},
            )
            tts_plan = BackendPlan(
                backend="matcha_rknn",
                provider="npu",
                env={"TTS_BACKEND": "matcha_rknn"},
            )

        elif mode == "interpret":
            # ASR + TTS parallel
            # ASR on CPU (sensevoice_sherpa: 50+ languages) + TTS on NPU (matcha_rknn)
            asr_plan = BackendPlan(
                backend="sensevoice_sherpa",
                provider="cpu",
                env={"ASR_BACKEND": "sensevoice_sherpa"},
            )
            tts_plan = BackendPlan(
                backend="matcha_rknn",
                provider="npu",
                env={"TTS_BACKEND": "matcha_rknn"},
            )

        elif mode == "asr_only":
            # Only ASR, all NPU to ASR
            asr_plan = BackendPlan(
                backend="qwen3_asr_rk",
                provider="npu",
                env={"ASR_BACKEND": "qwen3_asr_rk"},
            )
            tts_plan = None

        elif mode == "tts_only":
            # Only TTS
            asr_plan = None
            tts_plan = BackendPlan(
                backend="matcha_rknn",
                provider="npu",
                env={"TTS_BACKEND": "matcha_rknn"},
            )

        # Build result
        result = {
            "mode": mode,
            "description": mode_config["description"],
            "parallel": mode_config["parallel"],
            "asr": asr_plan.env if asr_plan else None,
            "tts": tts_plan.env if tts_plan else None,
            "warnings": warnings,
        }

        return result

    def get_mode_info(self, mode: str = None) -> dict:
        """Get information about a speech mode."""
        if mode is None:
            mode = os.environ.get("SPEECH_MODE", "dialogue")

        if mode not in self.MODES:
            return {
                "mode": mode,
                "valid": False,
                "description": None,
                "parallel": None,
            }

        return {
            "mode": mode,
            "valid": True,
            "description": self.MODES[mode]["description"],
            "parallel": self.MODES[mode]["parallel"],
        }
