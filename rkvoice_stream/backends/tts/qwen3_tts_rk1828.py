"""Qwen3-TTS on RK1828 (RKNN3 / PCIe coprocessor) TTSBackend.

Thin shell (``Qwen3TTSRK1828Backend``) over an IPC-backed service
(``Qwen3TTSRK1828Service``) that drives the on-device C++ demo through
``runtime.rknn3_worker.RKNN3Worker`` (subprocess, server mode).

Contract parity (spec §10): the C++ worker emits int16 LE PCM @ 24kHz; the
Service adapts it to the exact contract the package's other TTS backends
expose to the app:
  - ``synthesize`` -> ``tuple[bytes, dict]``  (WAV bytes, PCM_16 @ 24kHz)
  - ``synthesize_stream`` -> ``Iterator[tuple[np.ndarray, dict]]``
        (float32 audio in [-1, 1] @ 24kHz, meta dict)
  - ``get_sample_rate`` -> 24000

This mirrors ``qwen3_tts.TTSService`` / ``Qwen3RKNNBackend`` *in shape* but the
service is IPC-backed rather than in-process (the RKNN3 pipeline runs in C++).
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Iterator, List, Optional

import numpy as np

from rkvoice_stream.engine.tts import TTSBackend

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
_INT16_SCALE = 32768.0

# ── sentence splitting (long-text runaway protection) ─────────────────────
# Qwen3-TTS degrades / "runs away" (repeats, trails into hallucinated audio)
# on long single utterances. The orchestrator (app/dialogue.py) already
# chunks LLM token streams into sentences, but a *direct* caller of the
# backend (synthesize / synthesize_stream with a whole paragraph) bypasses
# that. So the Service splits here too, making every caller — not just the
# orchestrator — safe. Mirrors the CJK ≤~48-char-per-sentence convention used
# across the project's other TTS backends.
_MAX_SENTENCE_CHARS = 48
# Split *after* a sentence-ending punctuation mark (kept with the sentence).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；…\n])\s*")
# Soft break points used when a single punctuation-bounded segment is still
# longer than _MAX_SENTENCE_CHARS (long run-on with only commas / spaces).
_SOFT_BREAK_CHARS = "，,、；; 　"


def _split_for_tts(text: str, max_chars: int = _MAX_SENTENCE_CHARS) -> List[str]:
    """Split ``text`` into TTS-sized sentences (≤ ~``max_chars`` each).

    Two passes:
      1. Split on sentence-ending punctuation (kept with the preceding text).
      2. Any resulting segment still longer than ``max_chars`` is further
         broken at the last soft delimiter (comma / space) before the cap,
         falling back to a hard character-count cut if no delimiter exists
         (e.g. an unpunctuated CJK run).

    Short / already-bounded text returns a single-element list, so the common
    case (a short utterance) is a no-op passed straight through to the worker.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars and not _SENTENCE_SPLIT_RE.search(text):
        return [text]

    out: List[str] = []
    for seg in _SENTENCE_SPLIT_RE.split(text):
        seg = seg.strip()
        if not seg:
            continue
        while len(seg) > max_chars:
            # Prefer breaking at the last soft delimiter at/under the cap.
            cut = -1
            for i in range(min(max_chars, len(seg) - 1), 0, -1):
                if seg[i] in _SOFT_BREAK_CHARS:
                    cut = i + 1  # keep the delimiter with the left piece
                    break
            if cut <= 0:
                cut = max_chars  # no delimiter: hard cut at the cap
            piece = seg[:cut].strip()
            if piece:
                out.append(piece)
            seg = seg[cut:].strip()
        if seg:
            out.append(seg)
    return out


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert int16 LE PCM bytes to float32 audio in [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return samples / _INT16_SCALE


class Qwen3TTSRK1828Service:
    """IPC-backed Qwen3-TTS service driving the RK1828 C++ worker."""

    def __init__(
        self,
        binary_path: str,
        model_dir: str,
        ref_speaker: str = "girl_base",
        device_id: Optional[str] = None,
    ) -> None:
        self._binary_path = binary_path
        self._model_dir = model_dir
        self._ref_speaker = ref_speaker
        self._device_id = device_id
        self._worker = None

    def load(self) -> None:
        from rkvoice_stream.runtime.rknn3_worker import RKNN3Worker

        logger.info(
            "Starting RK1828 Qwen3-TTS worker: binary=%s model=%s speaker=%s device=%s",
            self._binary_path, self._model_dir, self._ref_speaker, self._device_id,
        )
        self._worker = RKNN3Worker(
            binary_path=self._binary_path,
            model_dir=self._model_dir,
            ref_speaker=self._ref_speaker,
            device_id=self._device_id,
        )
        self._worker.start()
        logger.info("RK1828 Qwen3-TTS service ready")

    def is_ready(self) -> bool:
        return self._worker is not None and self._worker.is_ready()

    def get_sample_rate(self) -> int:
        return SAMPLE_RATE

    def synthesize(self, text: str, **_kwargs) -> tuple[bytes, dict]:
        """Synthesize the whole utterance and return (WAV bytes, meta).

        Long text is split into ≤~48-char sentences (runaway protection) and
        each sentence is synthesized as a separate worker request; the PCM is
        concatenated back into one utterance for the caller.
        """
        if self._worker is None:
            raise RuntimeError("RK1828 TTS service not loaded — call load() first")

        t0 = time.perf_counter()
        sentences = _split_for_tts(text)
        if not sentences:
            pcm = b""
        else:
            pcm = b"".join(self._worker.synthesize(s) for s in sentences)
        audio = _pcm16_to_float32(pcm)
        elapsed = time.perf_counter() - t0

        duration = len(audio) / SAMPLE_RATE if len(audio) else 0.0
        wav_bytes = self._make_wav(audio)
        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0.0,
            "samples": int(len(audio)),
        }
        return wav_bytes, meta

    def synthesize_stream(self, text: str, **_kwargs) -> Iterator[tuple[np.ndarray, dict]]:
        """Stream float32 chunks as the worker emits int16 PCM frames.

        Long text is split into ≤~48-char sentences (runaway protection); each
        sentence is streamed sequentially through the worker so a long paragraph
        never becomes one runaway request. ``chunk_index`` is continuous across
        sentence boundaries; ``sentence_index`` marks which sentence a chunk
        belongs to.
        """
        if self._worker is None:
            raise RuntimeError("RK1828 TTS service not loaded — call load() first")

        t0 = time.perf_counter()
        chunk_index = 0
        for sentence_index, sentence in enumerate(_split_for_tts(text)):
            for pcm in self._worker.synthesize_stream(sentence):
                audio = _pcm16_to_float32(pcm)
                if len(audio) == 0:
                    continue
                meta = {
                    "chunk_index": chunk_index,
                    "sentence_index": sentence_index,
                    "samples": int(len(audio)),
                    "elapsed": round(time.perf_counter() - t0, 3),
                }
                chunk_index += 1
                yield audio, meta

    def cleanup(self) -> None:
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("RK1828 worker stop error: %s", exc)
            self._worker = None

    @staticmethod
    def _make_wav(audio: np.ndarray) -> bytes:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class Qwen3TTSRK1828Backend(TTSBackend):
    """Qwen3-TTS pipeline on the RK1828 PCIe coprocessor (RKNN3 via C++ worker)."""

    supports_streaming = True

    def __init__(self) -> None:
        self._service: Optional[Qwen3TTSRK1828Service] = None

    @property
    def name(self) -> str:
        return "qwen3_tts_rk1828"

    def is_ready(self) -> bool:
        return self._service is not None and self._service.is_ready()

    def preload(self) -> None:
        binary_path = os.environ.get("RK1828_TTS_BINARY", "/opt/rk1828/rknn_qwen3_tts_demo")
        model_dir = os.environ.get("RK1828_TTS_MODEL_DIR") or os.environ.get(
            "TTS_MODEL_DIR", "/opt/rk1828/models/qwen3_tts"
        )
        ref_speaker = os.environ.get("RK1828_TTS_REF_SPEAKER", "girl_base")
        device_id = os.environ.get("RK1828_DEVICE_ID") or None

        self._service = Qwen3TTSRK1828Service(
            binary_path=binary_path,
            model_dir=model_dir,
            ref_speaker=ref_speaker,
            device_id=device_id,
        )
        self._service.load()

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
        return self._service.synthesize(text)

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
        yield from self._service.synthesize_stream(text)

    def get_sample_rate(self) -> int:
        return SAMPLE_RATE

    def runtime_info(self) -> dict:
        info: dict = {"backend": self.name, "sample_rate": SAMPLE_RATE}
        if self._service is not None:
            info["device_id"] = self._service._device_id
            info["binary"] = self._service._binary_path
            info["model_dir"] = self._service._model_dir
            info["ref_speaker"] = self._service._ref_speaker
        return info

    def cleanup(self) -> None:
        if self._service is not None:
            try:
                self._service.cleanup()
            except Exception as exc:
                logger.warning("RK1828 TTS service cleanup error: %s", exc)
            self._service = None
