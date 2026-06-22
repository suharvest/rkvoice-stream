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
import struct

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RK3576 Speech Service", version="3.0.0")

_backend = None
_asr_backend = None
_audio_llm_backend = None
_dialogue = None
_resource_plan = None
_speech_mode = None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float | None = None
    pitch: float | None = None
    language: str | None = None  # e.g. "en_US", "zh_CN", "ja_JP", or None for auto-detect


@app.on_event("startup")
async def startup():
    global _backend, _asr_backend, _audio_llm_backend, _dialogue
    global _resource_plan, _speech_mode

    config_path = os.environ.get("CONFIG", "")
    if config_path:
        from pathlib import Path

        from rkvoice_stream import (
            _apply_asr_env,
            _apply_audio_llm_env,
            _apply_tts_env,
            load_config,
        )

        path = Path(config_path)
        if not path.exists() and not path.suffix:
            path = Path("configs") / f"{config_path}.yaml"
        config = load_config(str(path))
        if config.get("asr"):
            _apply_asr_env(config["asr"])
        if config.get("tts"):
            _apply_tts_env(config["tts"])
        if config.get("audio_llm"):
            _apply_audio_llm_env(config["audio_llm"])
        logger.info("Loaded runtime profile: %s", path)

    # --- ResourcePlanner: auto-select backends if SPEECH_MODE is set ---
    speech_mode = os.environ.get("SPEECH_MODE", "")
    tts_backend_name = os.environ.get("TTS_BACKEND", "")
    asr_backend_name = os.environ.get("ASR_BACKEND", "")
    audio_llm_backend_name = os.environ.get("AUDIO_LLM_BACKEND", "")
    require_tts_backend = _env_flag("REQUIRE_TTS_BACKEND")
    require_asr_backend = _env_flag("REQUIRE_ASR_BACKEND")
    require_audio_llm_backend = _env_flag("REQUIRE_AUDIO_LLM_BACKEND")

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
            if require_tts_backend:
                logger.error("Required TTS backend '%s' failed to load: %s", tts_backend_name, e)
                raise RuntimeError(f"Required TTS backend '{tts_backend_name}' failed to load") from e
            logger.error("Failed to load TTS backend '%s': %s — TTS disabled", tts_backend_name, e)
            _backend = None
    else:
        if require_tts_backend:
            raise RuntimeError("REQUIRE_TTS_BACKEND=1 but TTS_BACKEND is not set or disabled")
        logger.info("TTS_BACKEND not set or disabled — TTS disabled.")

    # --- ASR (optional) ---
    if asr_backend_name and asr_backend_name != "disabled":
        logger.info("Loading ASR backend: %s", asr_backend_name)
        try:
            from rkvoice_stream.engine.asr import create_asr_backend
            _asr_backend = create_asr_backend(asr_backend_name)
            _asr_backend.preload()
            logger.info("ASR backend '%s' ready.", _asr_backend.name)
        except Exception as e:
            if require_asr_backend:
                logger.error("Required ASR backend '%s' failed to load: %s", asr_backend_name, e)
                raise RuntimeError(f"Required ASR backend '{asr_backend_name}' failed to load") from e
            logger.error("Failed to load ASR backend '%s': %s — ASR disabled", asr_backend_name, e)
            _asr_backend = None
    else:
        if require_asr_backend:
            raise RuntimeError("REQUIRE_ASR_BACKEND=1 but ASR_BACKEND is not set or disabled")
        logger.info("ASR_BACKEND not set — ASR disabled.")

    # --- AudioLLM (optional; opt-in via AUDIO_LLM_BACKEND, e.g. gemma4_rk1828) ---
    # Audio -> text understanding model that collapses ASR+LLM (Phase 2). It does
    # NOT participate in SPEECH_MODE auto-selection — set AUDIO_LLM_BACKEND
    # explicitly. On a single RK1828 EP gemma4 and Qwen3-TTS are memory-exclusive
    # (~5GB); the V2V device placement (gemma4 on RK1828 + TTS on host, or
    # time-sharing) is a deployment decision — see capability.py device buckets.
    if audio_llm_backend_name and audio_llm_backend_name != "disabled":
        logger.info("Loading AudioLLM backend: %s", audio_llm_backend_name)
        try:
            from rkvoice_stream.engine.audio_llm import create_audio_llm
            _audio_llm_backend = create_audio_llm(audio_llm_backend_name)
            _audio_llm_backend.preload()
            logger.info("AudioLLM backend '%s' ready.", _audio_llm_backend.name)
        except Exception as e:
            if require_audio_llm_backend:
                logger.error("Required AudioLLM backend '%s' failed to load: %s", audio_llm_backend_name, e)
                raise RuntimeError(
                    f"Required AudioLLM backend '{audio_llm_backend_name}' failed to load"
                ) from e
            logger.error("Failed to load AudioLLM backend '%s': %s — AudioLLM disabled", audio_llm_backend_name, e)
            _audio_llm_backend = None
    else:
        if require_audio_llm_backend:
            raise RuntimeError("REQUIRE_AUDIO_LLM_BACKEND=1 but AUDIO_LLM_BACKEND is not set or disabled")
        logger.info("AUDIO_LLM_BACKEND not set — AudioLLM disabled.")

    # --- Dialogue orchestrator (requires TTS) ---
    if _backend:
        from rkvoice_stream.app.dialogue import DialogueOrchestrator
        _dialogue = DialogueOrchestrator(
            tts_backend=_backend,
            audio_llm_backend=_audio_llm_backend,
        )
        mode_note = "AudioLLM understanding" if _audio_llm_backend else "echo mode — no LLM"
        logger.info("Dialogue orchestrator ready (%s).", mode_note)


@app.on_event("shutdown")
async def shutdown():
    """Gracefully destroy NPU resources to prevent zombie threads."""
    global _backend, _asr_backend, _audio_llm_backend
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

    if _audio_llm_backend and hasattr(_audio_llm_backend, 'cleanup'):
        try:
            _audio_llm_backend.cleanup()
        except Exception as e:
            logger.warning("AudioLLM cleanup error: %s", e)

    logger.info("Shutdown complete.")


@app.get("/health")
async def health():
    from rkvoice_stream.engine.asr import ASRCapability
    asr_ready = _asr_backend.is_ready() if _asr_backend else False
    result = {
        "tts": _backend.is_ready() if _backend else False,
        "tts_backend": _backend.name if _backend and _backend.is_ready() else None,
        "streaming_tts": bool(_backend and getattr(_backend, "supports_streaming", False)),
        "asr": asr_ready,
        "asr_backend": _asr_backend.name if asr_ready else None,
        "streaming_asr": asr_ready and _asr_backend.has_capability(ASRCapability.STREAMING),
        "audio_llm": _audio_llm_backend.is_ready() if _audio_llm_backend else False,
        "audio_llm_backend": (
            _audio_llm_backend.name
            if _audio_llm_backend and _audio_llm_backend.is_ready()
            else None
        ),
    }
    if _audio_llm_backend and hasattr(_audio_llm_backend, "runtime_info"):
        try:
            info = _audio_llm_backend.runtime_info()
            if info:
                result["audio_llm_info"] = info
        except Exception as exc:
            result["audio_llm_info_error"] = str(exc)
    if _backend and hasattr(_backend, "runtime_info"):
        try:
            result["tts_info"] = _backend.runtime_info()
        except Exception as exc:
            result["tts_info_error"] = str(exc)
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

    Uses the backend streaming contract directly. MOSS emits the first decoded
    audio frame immediately, then may batch later codec frames to reduce steady
    state RTF while preserving low first-payload latency.
    """
    if not _backend or not _backend.is_ready():
        return JSONResponse({"error": "TTS not ready"}, status_code=503)
    if not getattr(_backend, "supports_streaming", False) or not hasattr(_backend, "synthesize_stream"):
        return JSONResponse({"error": "TTS backend does not support streaming"}, status_code=501)

    sr = _backend.get_sample_rate()

    async def stream():
        yield struct.pack("<I", sr)
        loop = asyncio.get_event_loop()

        q: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=2)

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
                q.put(exc)
            finally:
                q.put(None)

        loop.run_in_executor(None, _generate)
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is None:
                break
            if isinstance(chunk, BaseException):
                raise RuntimeError("TTS stream generation failed") from chunk
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
    prepare_task: asyncio.Task | None = None
    # Tracks the most recent archive_text we've already streamed as a final.
    # Used so VAD-triggered utterance finals get pushed to the client mid-session
    # without being re-emitted by the session-end final block.
    streamed_state = {"last_archive": ""}

    _last_partial: dict[str, str] = {"text": ""}

    async def _send_partial_if_any():
        try:
            partial, _ = stream.get_partial()
            if partial and partial != _last_partial["text"]:
                _last_partial["text"] = partial
                await ws.send_json({"text": partial, "is_final": False})
        except Exception:
            pass

    async def _maybe_emit_utterance_final() -> bool:
        """If the streaming class just produced a new VAD-triggered final,
        push it to the client right away. Returns True if anything was sent."""
        sess = getattr(stream, "_stream", None)
        if sess is None:
            return False
        archive = getattr(sess, "_archive_text", "") or ""
        ep_final = getattr(sess, "_episode_final", False)
        if not (ep_final and archive):
            return False
        if archive == streamed_state["last_archive"]:
            return False
        streamed_state["last_archive"] = archive
        try:
            await ws.send_json({
                "text": archive,
                "is_final": True,
                "session_complete": False,
            })
        except Exception:
            pass
        return True

    async def worker():
        while True:
            item = await audio_q.get()
            try:
                if item is None:
                    break
                sr, samples = item
                try:
                    await loop.run_in_executor(
                        None, stream.accept_waveform, sr, samples)
                except Exception as exc:
                    logger.debug("ASR worker accept_waveform error: %s", exc)
                    continue
                # Always check for VAD-triggered utterance finals, even after
                # EOU — multi-utterance clients need to see them.  Partials,
                # on the other hand, are skipped post-EOU (saves ws.send).
                if await _maybe_emit_utterance_final():
                    continue
                if eou_state["pending"]:
                    continue
                await _send_partial_if_any()
            finally:
                audio_q.task_done()

    async def prepare_when_drained():
        await audio_q.join()
        await loop.run_in_executor(None, stream.prepare_finalize)

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
            if mtype in {"prepare", "pre_eou", "prepare_finalize"}:
                # Frontend/external VAD can send this as soon as it enters
                # tail-silence.  It hides final encoder/decoder work under the
                # remaining VAD hangover, while the later EOU frame simply
                # drains the cached final text.
                eou_state["pending"] = True
                if prepare_task is None or prepare_task.done():
                    prepare_task = asyncio.create_task(prepare_when_drained())
                continue
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
            if prepare_task is not None:
                await prepare_task
            else:
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
            # Mark this as the session-end final.  Multi-utterance clients
            # use session_complete=False frames for VAD-triggered utterance
            # finals and session_complete=True to know the WS will close.
            already_streamed = (
                final_text and final_text == streamed_state["last_archive"])
            await ws.send_json({
                "text": final_text,
                "is_final": True,
                "session_complete": True,
                "duplicate_of_streamed": already_streamed,
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
         then backend-native int16 PCM chunks as soon as they are decoded.
      3. Server sends JSON: {"done": true, "chunks": N} when finished.
      4. Client can send another message or close.

    Requires TTS backend to be loaded. LLM is optional (defaults to echo mode).
    """
    await ws.accept()

    if not _dialogue:
        await ws.send_json({"error": "Dialogue not available (TTS not loaded)"})
        await ws.close()
        return
    if not _backend or not getattr(_backend, "supports_streaming", False) or not hasattr(_backend, "synthesize_stream"):
        await ws.send_json({"error": "Dialogue requires streaming TTS backend"})
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

            try:
                async for pcm_chunk in _dialogue.process_turn_pcm(user_text):
                    await ws.send_bytes(pcm_chunk)
                    chunk_count += 1
            except Exception as exc:
                logger.error("dialogue streaming failed: %s", exc)
                await ws.send_json({"error": "dialogue streaming failed", "chunks": chunk_count})
                continue

            await ws.send_json({"done": True, "chunks": chunk_count})

    except Exception as exc:
        logger.debug("Dialogue WS error: %s", exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio dialogue route (V2V: audio → AudioLLM (audio→text) → TTS → audio)
# ---------------------------------------------------------------------------

def _decode_audio_bytes(audio_bytes: bytes, fallback_sr: int = 16000):
    """Decode WAV/FLAC/… bytes (or raw int16 PCM) to (float32 mono, sample_rate).

    Tries a container decode via soundfile first; if that fails the payload is
    treated as raw little-endian int16 PCM at ``fallback_sr``.
    """
    import io as _io

    try:
        import soundfile as sf
        audio, sr = sf.read(_io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)
    except Exception:
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm, fallback_sr


@app.websocket("/audio_dialogue")
async def audio_dialogue_ws(ws: WebSocket):
    """Streaming V2V dialogue with an audio-in understanding stage.

    Collapses ASR + LLM into the AudioLLM backend (e.g. gemma4 on RK1828):
    client sends audio, the AudioLLM produces text, which is pipelined into
    streaming TTS and returned as PCM audio.

    Protocol:
      1. Client sends a binary frame: the input audio (WAV/FLAC bytes, or raw
         int16 PCM at 16 kHz). Optionally precede it with a JSON text frame
         ``{"prompt": "...", "sample_rate": 16000}`` to set the AudioLLM prompt
         and declare the sample rate for the raw-PCM case.
      2. Server streams binary: first 4 bytes = sample_rate (uint32 LE), then
         backend-native int16 PCM chunks as soon as they are decoded.
      3. Server sends JSON: {"done": true, "chunks": N}.
      4. Client can send another turn or close.

    Requires both an AudioLLM backend and a streaming TTS backend.
    """
    await ws.accept()

    if not _dialogue or not _dialogue.has_audio_llm():
        await ws.send_json({"error": "Audio dialogue not available (AudioLLM not loaded)"})
        await ws.close()
        return
    if not _backend or not getattr(_backend, "supports_streaming", False) or not hasattr(_backend, "synthesize_stream"):
        await ws.send_json({"error": "Audio dialogue requires streaming TTS backend"})
        await ws.close()
        return

    try:
        prompt: str | None = None
        sample_rate = 16000
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            text = msg.get("text")
            if text is not None:
                # Optional control frame: prompt / sample_rate for the next turn.
                try:
                    ctl = json.loads(text)
                except Exception:
                    await ws.send_json({"error": "bad control message"})
                    continue
                if "prompt" in ctl:
                    prompt = ctl.get("prompt") or None
                if "sample_rate" in ctl:
                    try:
                        sample_rate = int(ctl["sample_rate"])
                    except (TypeError, ValueError):
                        pass
                continue

            data = msg.get("bytes")
            if data is None or not data:
                continue

            audio, sr = _decode_audio_bytes(data, fallback_sr=sample_rate)
            logger.info("audio_dialogue: audio=%d samples @%dHz prompt=%r",
                        audio.size, sr, (prompt or "")[:40])
            chunk_count = 0
            try:
                async for pcm_chunk in _dialogue.process_audio_turn_pcm(
                    audio, sr, prompt=prompt
                ):
                    await ws.send_bytes(pcm_chunk)
                    chunk_count += 1
            except Exception as exc:
                logger.error("audio_dialogue streaming failed: %s", exc)
                await ws.send_json({"error": "audio dialogue streaming failed", "chunks": chunk_count})
                continue

            await ws.send_json({"done": True, "chunks": chunk_count})

    except Exception as exc:
        logger.debug("Audio dialogue WS error: %s", exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
