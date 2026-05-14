"""
Qwen3-ASR Runtime Library for Rockchip RK3576/RK3588

Encoder: RKNN NPU (Whisper-style mel→embedding)
Decoder: RKLLM NPU (Qwen3 LLM with EMBED input mode)

Usage:
    from qwen3asr import Qwen3ASREngine

    engine = Qwen3ASREngine("/path/to/models")
    result = engine.transcribe("audio.wav")
    print(result["text"])

    # Streaming
    stream = engine.create_stream()
    stream.feed_audio(pcm_chunk)
    print(stream.get_result())
    final = stream.finish()
"""

from .engine import Qwen3ASREngine
from .stream import StreamSession
from .chunk_confirm import ChunkConfirmASRStream
from .config import SUPPORTED_LANGUAGES, DEFAULT_CONFIG
from rkvoice_stream.vad.silero import SileroVAD

__version__ = "1.4.0"
__all__ = ["Qwen3ASREngine", "StreamSession", "ChunkConfirmASRStream",
           "SileroVAD", "SUPPORTED_LANGUAGES", "DEFAULT_CONFIG"]
