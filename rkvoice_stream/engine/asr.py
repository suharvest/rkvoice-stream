"""ASR backend abstraction for RK3576 speech service.

Mirrors the pattern from the Jetson version (app/asr_backend.py).
Select backend via ASR_BACKEND env var (default: qwen3_asr_rk).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRCapability(str, Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"
    MULTI_LANGUAGE = "multi_language"


class TranscriptionResult:
    def __init__(self, text: str, language: Optional[str] = None, **meta):
        self.text = text
        self.language = language
        self.meta = meta


class ASRStream(ABC):
    """A streaming ASR session that accumulates audio and produces text."""

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed audio samples (float32, [-1,1]) into the stream."""
        ...

    @abstractmethod
    def finalize(self) -> str:
        """Signal end-of-audio and return final transcription text."""
        ...

    def prepare_finalize(self) -> None:
        """Pre-encode remaining audio so finalize() only runs the decoder.

        Optional optimization — finalize() works without calling this first.
        """
        pass

    def get_partial(self) -> tuple[str, bool]:
        """Return (partial_text, is_endpoint). Default: no partial results."""
        return "", False


class ASRBackend(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> set[ASRCapability]: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def preload(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult: ...

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability."""
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        return cap in self.capabilities


def create_asr_backend(backend_name: Optional[str] = None) -> ASRBackend:
    if backend_name is None:
        backend_name = os.environ.get("ASR_BACKEND", "qwen3_asr_rk")

    if backend_name == "qwen3_asr_rk":
        from rkvoice_stream.backends.asr.qwen3_rk import Qwen3ASRRKBackend
        return Qwen3ASRRKBackend()
    elif backend_name == "paraformer_sherpa":
        from rkvoice_stream.backends.asr.paraformer_sherpa import ParaformerSherpaBackend
        return ParaformerSherpaBackend()
    elif backend_name == "sensevoice_sherpa":
        from rkvoice_stream.backends.asr.sensevoice_sherpa import SenseVoiceSherpaBackend
        return SenseVoiceSherpaBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name!r}")


# Public API alias
create_asr = create_asr_backend
