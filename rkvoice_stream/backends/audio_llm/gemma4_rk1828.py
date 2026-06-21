"""Gemma-4 on RK1828 (RKNN3 / PCIe coprocessor) AudioLLMBackend.

Thin shell (``Gemma4RK1828Backend``) over an IPC-backed service
(``Gemma4RK1828Service``) that drives the on-device gemma4 C++ demo (server
mode, Phase 2) through ``runtime.rknn3_worker.AudioLLMWorker``.

This mirrors the ``qwen3_tts_rk1828`` backend *in shape* (thin backend + IPC
Service over a subprocess worker) but speaks the Phase 2 text-token protocol
(spec §5): JSON request line in, length-prefixed UTF-8 text tokens out,
``END_OF_STREAM`` sentinel, ``READY <version>`` handshake.

The worker takes an ``audio_ref`` (a file path the C++ side reads); the Service
materialises the in-memory float32 audio to a temp WAV file so the package's
public ``generate(audio, sample_rate, ...)`` contract holds (the C++ binary
itself only deals in files / device buffers).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterator, Optional

import numpy as np

from rkvoice_stream.engine.audio_llm import AudioLLMBackend, DEFAULT_MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

# The gemma4 audio encoder expects 16kHz mono.
TARGET_SAMPLE_RATE = 16000


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear resample (avoids a hard scipy dependency)."""
    if src_sr == dst_sr or audio.size == 0:
        return audio
    n_dst = int(round(audio.size * dst_sr / src_sr))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class Gemma4RK1828Service:
    """IPC-backed Gemma-4 AudioLLM service driving the RK1828 C++ worker."""

    def __init__(
        self,
        binary_path: str,
        model_dir: str,
        device_id: Optional[str] = None,
    ) -> None:
        self._binary_path = binary_path
        self._model_dir = model_dir
        self._device_id = device_id
        self._worker = None

    def load(self) -> None:
        from rkvoice_stream.runtime.rknn3_worker import AudioLLMWorker

        logger.info(
            "Starting RK1828 Gemma-4 AudioLLM worker: binary=%s model=%s device=%s",
            self._binary_path, self._model_dir, self._device_id,
        )
        self._worker = AudioLLMWorker(
            binary_path=self._binary_path,
            model_dir=self._model_dir,
            device_id=self._device_id,
        )
        self._worker.start()
        logger.info("RK1828 Gemma-4 AudioLLM service ready")

    def is_ready(self) -> bool:
        return self._worker is not None and self._worker.is_ready()

    def _write_audio_ref(self, audio: np.ndarray, sample_rate: int) -> str:
        """Materialise audio to a temp 16kHz mono WAV; return its path."""
        import soundfile as sf

        mono = _to_mono_float32(audio)
        mono = _resample(mono, sample_rate, TARGET_SAMPLE_RATE)
        fd, path = tempfile.mkstemp(prefix="rk1828_gemma4_", suffix=".wav")
        os.close(fd)
        sf.write(path, mono, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return path

    def generate_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        if self._worker is None:
            raise RuntimeError("RK1828 AudioLLM service not loaded — call load() first")

        audio_ref = self._write_audio_ref(audio, sample_rate)
        try:
            for token in self._worker.generate_stream(
                audio_ref, prompt=prompt, max_new_tokens=max_new_tokens
            ):
                yield token
        finally:
            try:
                os.unlink(audio_ref)
            except OSError:
                pass

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        return "".join(
            self.generate_stream(audio, sample_rate, prompt, max_new_tokens)
        )

    def cleanup(self) -> None:
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("RK1828 AudioLLM worker stop error: %s", exc)
            self._worker = None


class Gemma4RK1828Backend(AudioLLMBackend):
    """Gemma-4 multimodal (audio -> text) on the RK1828 PCIe coprocessor."""

    def __init__(self) -> None:
        self._service: Optional[Gemma4RK1828Service] = None

    @property
    def name(self) -> str:
        return "gemma4_rk1828"

    def is_ready(self) -> bool:
        return self._service is not None and self._service.is_ready()

    def get_capabilities(self) -> set[str]:
        return {"audio"}

    def preload(self) -> None:
        binary_path = os.environ.get(
            "RK1828_GEMMA4_BINARY", "/opt/rk1828/rknn_gemma4_demo"
        )
        model_dir = os.environ.get(
            "RK1828_GEMMA4_MODEL_DIR", "/opt/rk1828/models/gemma4"
        )
        device_id = os.environ.get("RK1828_DEVICE_ID") or None

        self._service = Gemma4RK1828Service(
            binary_path=binary_path,
            model_dir=model_dir,
            device_id=device_id,
        )
        self._service.load()

    def generate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> str:
        if self._service is None:
            raise RuntimeError("Backend not loaded — call preload() first")
        return self._service.generate(audio, sample_rate, prompt, max_new_tokens)

    def generate_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> Iterator[str]:
        if self._service is None:
            raise RuntimeError("Backend not loaded — call preload() first")
        yield from self._service.generate_stream(
            audio, sample_rate, prompt, max_new_tokens
        )

    def runtime_info(self) -> dict:
        info: dict = {"backend": self.name, "capabilities": sorted(self.get_capabilities())}
        if self._service is not None:
            info["device_id"] = self._service._device_id
            info["binary"] = self._service._binary_path
            info["model_dir"] = self._service._model_dir
        return info

    def cleanup(self) -> None:
        if self._service is not None:
            try:
                self._service.cleanup()
            except Exception as exc:
                logger.warning("RK1828 AudioLLM service cleanup error: %s", exc)
            self._service = None
