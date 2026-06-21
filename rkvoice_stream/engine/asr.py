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

    prefer_backend_endpoint_vad: bool = False
    """Whether backend endpointing should own VAD finalization.

    Front-end VAD can still provide speech start/end events, but callers should
    wait for the backend's ``get_partial(..., is_endpoint=True)`` signal before
    finalizing when this flag is set.
    """

    allow_frontend_eou_finalize: bool = False
    """Whether frontend VAD speech_end may finalize this stream even when
    backend endpointing is preferred."""

    frontend_eou_min_audio_s: float = 0.0
    """Minimum accepted audio duration before frontend EOU can finalize a
    backend-owned endpoint stream."""

    immediate_client_eos_cancel_safe: bool = False
    """Whether partial abort can run outside normal ASR serialization."""

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

    def cancel_and_finalize(self) -> None:
        """Hard-cancel pending partial decodes and skip any residual tail
        encoding, so the subsequent ``finalize()`` returns as fast as
        possible.  Used by the WebSocket EOU control message.

        Default: no-op (legacy backends just run finalize directly).
        """
        pass

    def abort_partial_decode(self) -> None:
        """Best-effort abort of an in-flight partial decode only.

        This must not enter finalizing mode or drop future audio. Backends that
        cannot distinguish partial abort from final cancel should leave the
        default no-op in place.
        """
        pass

    def get_partial(self) -> tuple[str, bool]:
        """Return (partial_text, is_endpoint). Default: no partial results."""
        return "", False


class OfflineAccumulateStream(ASRStream):
    """Generic offline→streaming adapter.

    Wraps any offline backend (one implementing ``transcribe_array``) into a
    streaming session: accumulate audio, transcribe the whole utterance on
    ``finalize()``. Endpointing is delegated to the OVS server-side VAD (which
    finalizes + recreates the stream per utterance), so there is NO internal VAD
    and NO incremental partial. Any backend that sets
    ``supports_offline_streaming = True`` gets this for free via
    ``ASRBackend.create_stream`` — no per-backend stream code.
    """

    def __init__(self, backend: "ASRBackend", language: str = "auto") -> None:
        self._backend = backend
        self._language = language
        self._buf: list = []

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self._buf.append(np.asarray(samples, dtype=np.float32))

    def get_partial(self) -> tuple[str, bool]:
        return "", False

    def finalize(self) -> str:
        if not self._buf:
            return ""
        audio = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        self._buf = []
        return self._backend.transcribe_array(audio, self._language).text

    def close(self) -> None:
        self._buf = []


class ASRBackend(ABC):
    prefer_backend_endpoint_vad: bool = False
    """Whether streams from this backend should receive audio before frontend
    VAD speech_start and rely on backend endpointing for finalization."""

    # Offline backends set this True + implement transcribe_array() to get a
    # streaming session (via OfflineAccumulateStream) + the STREAMING capability
    # for free — no per-backend stream code, no internal VAD.
    supports_offline_streaming: bool = False

    allow_frontend_eou_finalize: bool = False
    """Whether frontend VAD speech_end may finalize streams from this backend
    even when backend endpointing is preferred."""

    frontend_eou_min_audio_s: float = 0.0
    """Minimum accepted audio duration before frontend EOU can finalize."""

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

    def transcribe_array(self, samples: np.ndarray, language: str = "auto") -> TranscriptionResult:
        """One-shot offline transcription on a float32 mono 16k sample array.

        Implemented by offline backends that opt into ``supports_offline_streaming``;
        the generic OfflineAccumulateStream calls this on finalize().
        """
        raise NotImplementedError(f"{self.name} does not implement transcribe_array")

    def create_stream(
        self,
        language: str = "auto",
        stream_options: Optional[dict] = None,
    ) -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability.

        Offline backends with ``supports_offline_streaming`` get a generic
        accumulate-then-transcribe stream for free. ``stream_options`` is optional
        and session-scoped; legacy backends ignore it.
        """
        if self.supports_offline_streaming:
            return OfflineAccumulateStream(self, language)
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        if cap in self.capabilities:
            return True
        if cap == ASRCapability.STREAMING and self.supports_offline_streaming:
            return True
        return False


def create_asr_backend(backend_name: Optional[str] = None) -> ASRBackend:
    if backend_name is None:
        backend_name = os.environ.get("ASR_BACKEND", "qwen3_asr_rk")

    if backend_name == "qwen3_asr_rk":
        from rkvoice_stream.backends.asr.qwen3_rk import Qwen3ASRRKBackend
        return Qwen3ASRRKBackend()
    elif backend_name == "paraformer_rknn":
        from rkvoice_stream.backends.asr.paraformer_rknn import ParaformerRKNNBackend
        return ParaformerRKNNBackend()
    elif backend_name == "paraformer_sherpa":
        from rkvoice_stream.backends.asr.paraformer_sherpa import ParaformerSherpaBackend
        return ParaformerSherpaBackend()
    elif backend_name == "sensevoice_sherpa":
        from rkvoice_stream.backends.asr.sensevoice_sherpa import SenseVoiceSherpaBackend
        return SenseVoiceSherpaBackend()
    elif backend_name == "sensevoice_rknn":
        from rkvoice_stream.backends.asr.sensevoice_rknn import SenseVoiceRKNNBackend
        return SenseVoiceRKNNBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name!r}")


# Public API alias
create_asr = create_asr_backend
