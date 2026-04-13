"""TTS backend abstraction for RK3576.

Provides a simple abstract base class for TTS backends.
Select backend via TTS_BACKEND env var.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np


class TTSBackend(ABC):
    """Base class for all RK3576 TTS backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'qwen3_rknn')."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once at startup."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
        ...

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """Stream TTS, yielding (audio_float32_chunk, metadata) tuples.

        Default implementation falls back to synthesize() and yields one chunk.
        Override for true streaming.
        """
        import io
        import soundfile as sf

        wav_bytes, meta = self.synthesize(
            text=text, speaker_id=speaker_id,
            speed=speed, pitch_shift=pitch_shift, **kwargs
        )
        buf = io.BytesIO(wav_bytes)
        audio, _ = sf.read(buf, dtype="float32")
        yield audio, meta

    @abstractmethod
    def get_sample_rate(self) -> int:
        ...


def create_backend(backend_name: Optional[str] = None) -> TTSBackend:
    """Factory: create TTS backend by name.

    Args:
        backend_name: 'qwen3_rknn' or None for auto-detect from TTS_BACKEND env.
    """
    import os

    if backend_name is None:
        backend_name = os.environ.get("TTS_BACKEND", "qwen3_rknn")

    if backend_name == "qwen3_rknn":
        from rkvoice_stream.backends.tts.qwen3_rknn import Qwen3RKNNBackend
        return Qwen3RKNNBackend()
    elif backend_name == "matcha_rknn":
        from rkvoice_stream.backends.tts.matcha import MatchaRKNNBackend
        return MatchaRKNNBackend()
    elif backend_name == "piper_rknn":
        from rkvoice_stream.backends.tts.piper import PiperRKNNBackend
        return PiperRKNNBackend()
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name!r}")


# Public API alias
create_tts = create_backend
