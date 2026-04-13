"""Qwen3-TTS RKNN/RKLLM backend for RK3576.

Wraps the existing qwen3_tts.py TTSService into the TTSBackend interface.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

import numpy as np

from rkvoice_stream.engine.tts import TTSBackend

logger = logging.getLogger(__name__)


class Qwen3RKNNBackend(TTSBackend):
    """Qwen3-TTS pipeline using RKNN NPU + RKLLM talker on RK3576."""

    def __init__(self):
        self._service = None

    @property
    def name(self) -> str:
        return "qwen3_rknn"

    def is_ready(self) -> bool:
        return self._service is not None and self._service.is_ready()

    def preload(self) -> None:
        from rkvoice_stream.backends.tts.qwen3_tts import TTSService

        model_dir = os.environ.get("MODEL_DIR", "/opt/tts/models")
        logger.info("Loading Qwen3-TTS RKNN models from %s", model_dir)
        self._service = TTSService(model_dir)
        self._service.load()
        logger.info("Qwen3-TTS RKNN backend ready")

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._service is None:
            raise RuntimeError("Backend not loaded — call preload() first")

        # Serialize NPU access with ASR backend if it is loaded
        try:
            from rkvoice_stream.backends.asr.qwen3_rk import get_npu_lock
            lock = get_npu_lock()
        except ImportError:
            lock = None

        if lock is not None:
            with lock:
                return self._service.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    speed=speed or 1.0,
                )
        else:
            return self._service.synthesize(
                text=text,
                speaker_id=speaker_id,
                speed=speed or 1.0,
            )

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if self._service is None:
            raise RuntimeError("Backend not loaded — call preload() first")

        # Serialize NPU access with ASR backend if it is loaded
        try:
            from rkvoice_stream.backends.asr.qwen3_rk import get_npu_lock
            lock = get_npu_lock()
        except ImportError:
            lock = None

        if lock is not None:
            with lock:
                yield from self._service.synthesize_stream(
                    text=text,
                    speaker_id=speaker_id,
                    speed=speed or 1.0,
                )
        else:
            yield from self._service.synthesize_stream(
                text=text,
                speaker_id=speaker_id,
                speed=speed or 1.0,
            )

    def get_sample_rate(self) -> int:
        from rkvoice_stream.backends.tts.qwen3_tts import SAMPLE_RATE
        return SAMPLE_RATE

    def cleanup(self) -> None:
        """Release RKNN/RKLLM resources."""
        if self._service is not None:
            try:
                self._service.cleanup()
            except Exception as e:
                logger.warning("TTSService cleanup error: %s", e)
            self._service = None
