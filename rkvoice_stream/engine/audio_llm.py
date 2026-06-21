"""AudioLLM backend abstraction for Rockchip speech service.

An AudioLLM is a multimodal model that takes audio (plus an optional text
prompt) and produces text — collapsing the ASR + LLM "understanding" steps of a
V2V pipeline into one model (e.g. Gemma-4 on RK1828).  This neither fits the ASR
ABC (pure transcription) nor the TTS ABC (text -> audio), so it gets its own
engine abstraction.

Style: input is float32 mono audio + sample rate (mirrors ``engine/asr.py``);
streaming yields text tokens (mirrors ``engine/tts.py``'s streaming + factory
pattern).  Select backend via the ``AUDIO_LLM_BACKEND`` env var.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np

# Default token budget for a single generation.
DEFAULT_MAX_NEW_TOKENS = 256


class AudioLLMBackend(ABC):
    """Base class for all AudioLLM (audio -> text) backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'gemma4_rk1828')."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once at startup."""
        ...

    @abstractmethod
    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> str:
        """One-shot: consume audio (+ optional prompt), return the full text."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> Iterator[str]:
        """Stream text tokens as they are produced."""
        ...

    @abstractmethod
    def get_capabilities(self) -> set[str]:
        """Modalities/features this backend supports.

        At minimum ``{"audio"}``; reserved future values: ``"vision"``,
        ``"multiturn"``.
        """
        ...

    def runtime_info(self) -> dict:
        """Structured non-secret runtime metadata for /health."""
        return {}


def create_audio_llm(backend_name: Optional[str] = None) -> AudioLLMBackend:
    """Factory: create an AudioLLM backend by name.

    Args:
        backend_name: backend name, or None to read ``AUDIO_LLM_BACKEND`` env
            (default: ``gemma4_rk1828``).
    """
    if backend_name is None:
        backend_name = os.environ.get("AUDIO_LLM_BACKEND", "gemma4_rk1828")

    if backend_name == "gemma4_rk1828":
        from rkvoice_stream.backends.audio_llm.gemma4_rk1828 import Gemma4RK1828Backend
        return Gemma4RK1828Backend()
    else:
        raise ValueError(f"Unknown AudioLLM backend: {backend_name!r}")
