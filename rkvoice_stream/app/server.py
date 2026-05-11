"""FastAPI TTS+ASR service for RK3576 with pluggable backends.

Select backend via TTS_BACKEND / ASR_BACKEND env vars, or use SPEECH_MODE for auto-selection.

API-compatible with jetson-voice:
   POST /tts         — JSON {"text": "...", "sid": 0, "speed": 1.0} -> WAV
   POST /tts/stream  — streaming TTS (PCM chunks)
   POST /asr         — multipart upload -> {"text": ..., "language": ...}
   WS   /asr/stream  — streaming ASR (int16 PCM frames -> JSON)
   WS   /dialogue    — streaming dialogue (text in -> PCM audio chunks out)
   GET  /health      — health check
   GET  /mode        — current speech mode and plan
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import signal
import struct
import sys

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _signal_handler(signum, frame):
    logger.info("Signal %d received, shutting down...", signum)
    # FastAPI shutdown event will handle NPU resource cleanup
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

app = FastAPI(title="RK3576 Speech Service", version="3.0.0")

_backend = None
_asr_backend = None
_dialogue = None
_resource_plan = None
_speech_mode = None


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None  # e.g. "en_US", "zh_CN", "ja_JP", or None for auto-detect


@app.on_event("startup")
async def startup():
    global _backend, _asr_backend, _dialogue, _resource_plan, _speech_mode

    # --- ResourcePlanner: auto-select backends if SPEECH_MODE is set ---
    speech_mode = os.environ.get("SPEECH_MODE", "")
    tts_backend_name = os.environ.get("TTS_BACKEND", "")
    asr_backend_name = os.environ.get("ASR_BACKEND", "")

    if speech_mode and not tts_backend_name and not asr_backend_name:
        from rkvoice_stream.app.resource_planner import ResourcePlanner
        planner = ResourcePlanner(speech_mode)
        _resource_plan = planner.plan()
        _speech_mode = speech_mode
        logger.info("ResourcePlanner: mode=%s", speech_mode)
        if _resource_plan.get("asr"):
            asr_backend_name = _resource_plan["asr"]["backend"]
            logger.info("  ASR: %s (%s)", asr_backend_name, _resource_plan["asr"]["provider"])
        if _resource_plan.get("tts"):
            tts_backend_name = _resource_plan["tts"]["backend"]
            logger.info("  TTS: %s (%s)", tts_backend_name, _resource_plan["tts"]["provider"])
    else:
        _speech_mode = "custom" if (tts_backend_name or asr_backend_name) else None

    # --- TTS (optional) ---
    if tts_backend_name and tts_backend_name != "disabled":
        logger.info("Loading TTS backend: %s", tts_backend_name)
        try:
            from rkvoice_stream.engine.tts import create_backend
            _backend = create_backend(tts_backend_name)
            _backend.preload()
            logger.info("TTS backend '%s' ready.", _backend.name)
        except Exception as e:
            logger.error("Failed to load TTS backend '%s': %s — TTS disabled", tts_backend_name, e)
            _backend = None
    else:
        logger.info("TTS_BACKEND not set or disabled — TTS disabled.")

    # --- ASR (optional) ---
    asr_backend_name = os.environ.get("ASR_BACKEND", "")
    if asr_backend_name and asr_backend_name != "disabled":
        logger.info("Loading ASR backend: %s", asr_backend_name)
        try:
            from rkvoice_stream.engine.asr import create_asr_backend
            _asr_backend = create_asr_backend(asr_backend_name)
            _asr_backend.preload()
            logger.info("ASR backend '%s' ready.", _asr_backend.name)
        except Exception as e:
            logger.error("Failed to load ASR backend '%s': %s — ASR disabled", asr_backend_name, e)
            _asr_backend = None
    else:
        logger.info("ASR_BACKEND not set — ASR disabled.")

    # --- Dialogue orchestrator (requires TTS) ---
    if _backend:
        from rkvoice_stream.app.dialogue import DialogueOrchestrator
        _dialogue = DialogueOrchestrator(tts_backend=_backend)
        logger.info("Dialogue orchestrator ready (echo mode — no LLM).")


@app.on_event("shutdown")
async def shutdown():
    """Gracefully destroy NPU resources to prevent zombie threads."""
    global _backend, _asr_backend
    logger.info("Shutting down — releasing NPU resources...")

    if _backend and hasattr(_backend, 'cleanup'):
        try:
            _backend.cleanup()
        except Exception as e:
            logger.warning("TTS cleanup error: %s", e)

    if _asr_backend and hasattr(_asr_backend, 'cleanup'):
        try:
            _asr_backend.cleanup()
        except Exception as e:
            logger.warning("ASR cleanup error: %s", e)

    logger.info("Shutdown complete.")


@app.get("/health")
async def health():
    from rkvoice_stream.engine.asr import ASRCapability
    asr_ready = _asr_backend.is_ready() if _asr_backend else False
    result = {
        "tts": _backend.is_ready() if _backend else False,
        "tts_backend": _backend.name if _backend and _backend.is_ready() else None,
        "asr": asr_ready,
        "asr_backend": _asr_backend.name if asr_ready else None,
        "streaming_asr": asr_ready and _asr_backend.has_capability(ASRCapability.STREAMING),
    }
    if _speech_mode:
        result["mode"] = _speech_mode
    return result


@app.get("/mode")
async def mode():
    """Current speech mode and resource plan."""
    return {
        "mode": _speech_mode,
        "plan": _resource_plan,
        "available_modes": ["dialogue", "interpret", "asr_only", "tts_only"],
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    if not _backend or not _backend.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    loop = asyncio.get_event_loop()
    wav_bytes, meta = await loop.run_in_executor(
        None,
        lambda: _backend.synthesize(
            text=req.text,
            speaker_id=req.sid or 0,
            speed=req.speed,
            pitch_shift=req.pitch,
            language=req.language,
        ),
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta["duration"]),
            "X-Inference-Time": str(meta["inference_time"]),
            "X-RTF": str(meta["rtf"]),
        },
    )


@app.options("/tts/stream")
async def tts_stream_options():
    """Allow clients to probe for streaming support."""
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks.

    Uses real streaming: yields audio as soon as the first 10 AR frames are vocoded
    (~1.3s AR + ~250ms vocoder ≈ 1.6s TTFT), then continues in 25-frame chunks.
    """
    if not _backend or not _backend.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)

    sr = _backend.get_sample_rate()

    async def stream():
        yield struct.pack("<I", sr)
        loop = asyncio.get_event_loop()

        q: queue.Queue[bytes | None] = queue.Queue()

        def _generate():
            try:
                for audio_chunk, meta in _backend.synthesize_stream(
                    text=req.text,
                    speaker_id=req.sid or 0,
                    speed=req.speed,
                    pitch_shift=req.pitch,
                    language=req.language,
                ):
                    pcm = (np.clip(audio_chunk * 32767, -32768, 32767)
                           .astype(np.int16))
                    q.put(pcm.tobytes())
            except Exception as exc:
                logger.error("TTS stream generation error: %s", exc)
            finally:
                q.put(None)

        loop.run_in_executor(None, _generate)
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# ASR routes
# ---------------------------------------------------------------------------

@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    """Transcribe an audio file (WAV, FLAC, MP3, …).

    Returns JSON: {"text": "...", "language": "...", "backend": "..."}
    """
    if not _asr_backend or not _asr_backend.is_ready():
        return JSONResponse({"error": "ASR not ready"}, status_code=503)

    audio_bytes = await file.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _asr_backend.transcribe(audio_bytes, language=language),
    )
    return {
        "text": result.text,
        "language": result.language,
        "backend": _asr_backend.name,
        **result.meta,
    }


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
):
    """Streaming ASR over WebSocket.

    Client sends raw int16 PCM frames (at `sample_rate`).
    Server sends JSON objects: {"text": "...", "is_final": bool}.

    To finalize, the client may either:
      * Send an empty binary frame (legacy protocol). Server runs
        ``prepare_finalize`` (encodes any residual tail audio) then
        ``finalize``.
      * Send a JSON text message ``{"type": "eou"}``. Server calls
        ``cancel_and_finalize`` (drops residual sub-chunk audio, aborts
        any in-flight partial decode) then ``finalize`` — used to hit
        sub-500 ms stop→final when the dialogue manager already has its
        own upstream VAD and can declare end-of-utterance authoritatively.
      * Close the connection.
    """
    await ws.accept()

    if not _asr_backend or not _asr_backend.is_ready():
        await ws.send_json({"error": "ASR not ready"})
        await ws.close()
        return

    loop = asyncio.get_event_loop()
    stream = _asr_backend.create_stream(language=language)
    eou_fast_path = False  # True ⇒ skip prepare_finalize (already cancelled)

    # ── Worker thread + queue ───────────────────────────────────────────
    # Audio frames are pushed to an asyncio.Queue and consumed by a single
    # worker task that runs accept_waveform in the default executor.  This
    # decouples WS receive from the per-chunk processing latency so the
    # client's end-of-utterance control message (text frame) is read
    # immediately, even when the worker is mid-way through a VAD-triggered
    # 1.3-s final decode in the executor.
    audio_q: asyncio.Queue = asyncio.Queue()
    eou_state = {"pending": False}

    async def _send_partial_if_any():
        try:
            partial, _ = stream.get_partial()
            if partial:
                await ws.send_json({"text": partial, "is_final": False})
        except Exception:
            pass

    async def worker():
        while True:
            item = await audio_q.get()
            if item is None:
                break
            sr, samples = item
            try:
                await loop.run_in_executor(
                    None, stream.accept_waveform, sr, samples)
            except Exception as exc:
                logger.debug("ASR worker accept_waveform error: %s", exc)
                continue
            # Skip sending partials once EOU is signalled (saves a ws.send
            # round-trip per remaining chunk) but still process them so the
            # stream's internal buffer reflects the full client audio.
            if not eou_state["pending"]:
                await _send_partial_if_any()

    worker_task = asyncio.create_task(worker())

    try:
        while True:
            msg = await ws.receive()
            # Disconnect / close frame
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if not data:
                    # Empty binary frame signals end of audio (legacy).
                    break
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_q.put_nowait((sample_rate, samples))
                continue

            text = msg.get("text")
            if text is None:
                continue
            # JSON control message.
            try:
                ctl = json.loads(text)
            except Exception:
                logger.debug("ASR stream: bad control msg %r", text[:80])
                continue
            mtype = (ctl.get("type") or "").lower()
            if mtype == "eou":
                # Mark EOU so worker stops sending partials, then break to
                # finalize.  All audio queued so far will still be processed
                # by the worker (drained in `finally`).
                # We deliberately do NOT abort the decoder here — that
                # would truncate any in-flight VAD-triggered final decode.
                eou_state["pending"] = True
                eou_fast_path = True
                break
            if mtype == "stop":
                break
            logger.debug("ASR stream: unknown control type %r", mtype)

    except Exception as exc:
        logger.debug("ASR stream error: %s", exc)

    finally:
        # Signal worker to stop and wait for it to drain.
        try:
            audio_q.put_nowait(None)
        except Exception:
            pass
        # For b"" legacy path we want the worker to fully drain any queued
        # audio (so the trailing silence chunks are seen by the VAD).
        # For EOU fast path the worker will skip remaining items anyway.
        try:
            await asyncio.wait_for(worker_task, timeout=15.0)
        except Exception:
            worker_task.cancel()
        try:
            # Both legacy (b"") and EOU paths run prepare_finalize so the
            # last sub-chunk audio is encoded.  With the worker-queue model
            # all queued audio is processed before we reach this point, so
            # there's no need for cancel_and_finalize's drop-tail behavior.
            await loop.run_in_executor(None, stream.prepare_finalize)
            final_result = await loop.run_in_executor(None, stream.finalize)
            if isinstance(final_result, dict):
                final_text = final_result.get("text", "")
                final_meta = {
                    "final_mode": final_result.get("final_mode"),
                    "fallback": final_result.get("fallback"),
                    "finalize_ms": final_result.get("finalize_ms"),
                }
            else:
                final_text = final_result
                final_meta = {}
            await ws.send_json({
                "text": final_text,
                "is_final": True,
                **final_meta,
            })
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dialogue route (streaming pipeline: text → LLM → TTS → audio)
# ---------------------------------------------------------------------------

@app.websocket("/dialogue")
async def dialogue_ws(ws: WebSocket):
    """Streaming dialogue: client sends text, server streams TTS audio back.

    Protocol:
      1. Client sends JSON: {"text": "用户说的话"}
      2. Server streams binary: first 4 bytes = sample_rate (uint32 LE),
         then int16 PCM chunks (one per sentence).
      3. Server sends JSON: {"done": true, "chunks": N} when finished.
      4. Client can send another message or close.

    Requires TTS backend to be loaded. LLM is optional (defaults to echo mode).
    """
    await ws.accept()

    if not _dialogue:
        await ws.send_json({"error": "Dialogue not available (TTS not loaded)"})
        await ws.close()
        return

    try:
        while True:
            msg = await ws.receive_json()
            user_text = msg.get("text", "")
            if not user_text:
                await ws.send_json({"error": "empty text"})
                continue

            logger.info("dialogue: user=%r", user_text[:80])
            chunk_count = 0

            async for pcm_chunk in _dialogue.process_turn_pcm(user_text):
                await ws.send_bytes(pcm_chunk)
                chunk_count += 1

            await ws.send_json({"done": True, "chunks": chunk_count})

    except Exception as exc:
        logger.debug("Dialogue WS error: %s", exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
