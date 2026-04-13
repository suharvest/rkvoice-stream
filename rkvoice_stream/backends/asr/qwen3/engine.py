"""
Qwen3-ASR Engine: RKNN encoder + RKLLM/Matmul decoder pipeline for RK3576/RK3588.

Provides:
  - ``transcribe()``: Batch transcription with sliding-window memory.
  - ``create_stream()``: Streaming session (see ``stream.StreamSession``).

Decoder options:
  - RKLLM (default): Uses librkllmrt.so, ~16ms/token
  - Matmul: Uses RKNN matmul API, ~16ms/token, no RKLLM/RKNN conflict
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from .config import SAMPLE_RATE
from .encoder import RknnEncoder
from .utils import load_audio
from .stream import StreamSession


class Qwen3ASREngine:
    """
    Complete Qwen3-ASR engine for RK3576/RK3588.
    
    Combines RKNN encoder (NPU) and RKLLM decoder (NPU) for speech recognition.
    
    Directory layout expected:
        model_dir/
        ├── encoder/           (FP16 encoder models)
        │   └── {platform}/
        │       ├── *frontend*.rknn
        │       └── *backend*.rknn
        ├── encoder_int8/      (INT8 encoder models, optional)
        │   └── {platform}/
        │       ├── *frontend*.int8.*.rknn
        │       └── *backend*.int8.*.rknn
        ├── decoder/
        │   └── *.rkllm          (or under rkllm/)
        ├── mel_filters.npy
        ├── embed_tokens.npy     (or under embd/)
        └── tokenizer.json       (or auto-detected)
    
    Usage:
        engine = Qwen3ASREngine("/path/to/models")
        result = engine.transcribe("audio.wav", language="Chinese")
        engine.close()
    """

    def __init__(self, model_dir: str, platform: str = "rk3576",
                 lib_path: str = None,
                 tokenizer_path: str = None,
                 decoder_quant: str = "w4a16_g128",
                 encoder_quant: str = "fp16",
                 encoder_sizes: list = None,
                 npu_core_mask: str = "NPU_CORE_0_1",
                 enabled_cpus: int = 2,
                 max_context_len: int = 4096,
                 max_new_tokens: int = 100,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repeat_penalty: float = 1.15,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 embed_flash: int = 1,
                 compact_suffix: bool = True,
                 decoder_type: str = "rkllm",
                 decoder_exec_mode: str = "dual_core",
                 decoder_callback: Callable = None,
                 verbose: bool = True):
        """
        Initialize the ASR engine.

        Args:
            model_dir: Root model directory
            platform: "rk3576" or "rk3588"
            lib_path: Path to librkllmrt.so (auto-detected if None)
            tokenizer_path: Path to tokenizer.json (auto-detected if None)
            decoder_quant: Quantization type for decoder ("w4a16" or "w8a8")
            encoder_quant: Encoder quantization ("fp16" or "int8")
            encoder_sizes: List of encoder size variants to load (e.g. [2, 4]).
                           None = load all available sizes.
            enabled_cpus: CPU cores for RKLLM (2 or 4)
            max_context_len: Maximum KV cache context length
            max_new_tokens: Maximum tokens to generate per run
            top_k: Top-K sampling (1 = greedy, recommended for ASR)
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            repeat_penalty: Repetition penalty
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            embed_flash: Flash embedding mode (1 = enabled)
            compact_suffix: Use compact suffix prompt (saves ~120ms)
            decoder_type: "rkllm" (default) or "matmul" (no RKLLM conflict)
            decoder_exec_mode: "single_core" or "dual_core" (matmul only)
            decoder_callback: Optional callback(text, is_finish) for streaming
            verbose: Print loading progress
        """
        t_start = time.time()
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.platform = platform
        self.verbose = verbose
        self.compact_suffix = compact_suffix

        if verbose:
            print(f"=== Qwen3-ASR Engine v1.3.0 ===")
            print(f"Model dir: {model_dir}")
            print(f"Platform: {platform}")
            print(f"Encoder: {encoder_quant}")

        # ---- Auto-detect model files ----
        if encoder_quant == "int8":
            encoder_dir = model_dir / "encoder_int8" / platform
            if not encoder_dir.exists():
                print(f"  [WARN] INT8 encoder dir not found: {encoder_dir}")
                print(f"  Falling back to FP16 encoder")
                encoder_dir = model_dir / "encoder" / platform
                encoder_quant = "fp16"
        else:
            encoder_dir = model_dir / "encoder" / platform
        # Encoder dir may not exist if only merged models are present
        # in the parent (model_dir).  The encoder class scans upward.
        if not encoder_dir.exists():
            encoder_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  [INFO] Created encoder dir: {encoder_dir} "
                      f"(merged models expected in {model_dir})")

        mel_path = self._find_file(model_dir, "mel_filters.npy",
                                   "mel filters")

        # VAD model (optional)
        vad_path = model_dir / "vad" / "silero_vad.onnx"
        self.vad_model_path = str(vad_path) if vad_path.exists() else None
        if self.vad_model_path and verbose:
            print(f"  VAD model: {vad_path}")

        # Decoder model: check decoder/ then rkllm/ directory (only for rkllm type)
        decoder_dir = model_dir / "decoder"
        if not decoder_dir.exists():
            decoder_dir = model_dir / "rkllm"
        if decoder_type != "matmul":
            decoder_path = self._find_file(
                decoder_dir, f"*qwen3*{decoder_quant}*{platform}*rkllm",
                "decoder model", required=False
            )
            if decoder_path is None:
                decoder_path = self._find_file(
                    decoder_dir, f"*{decoder_quant}*{platform}*rkllm",
                    "decoder model"
                )
        else:
            decoder_path = None

        # Embedding table: check embed_tokens.npy or embd/ directory
        embd_path = model_dir / "embed_tokens.npy"
        if not embd_path.exists():
            embd_dir = model_dir / "embd"
            embd_path = self._find_file(embd_dir, "*embed_tokens*npy",
                                        "embedding table")

        # Tokenizer: auto-detect
        if tokenizer_path is None:
            tokenizer_path = self._find_tokenizer(model_dir)

        # Lib path (only needed for rkllm decoder)
        if decoder_type != "matmul" and lib_path is None:
            lib_dir = model_dir.parent / "lib"
            if not lib_dir.exists():
                lib_dir = model_dir / ".." / "lib"
            lib_path = self._find_file(lib_dir, "librkllmrt*so*",
                                       "RKLLM runtime library")

        # ---- Load components ----
        if verbose:
            print(f"\nLoading encoder...")
        self.encoder = RknnEncoder(
            str(encoder_dir), str(mel_path),
            npu_core_mask=npu_core_mask,
            sizes=encoder_sizes,
        )
        self.max_chunk_seconds = self.encoder.max_seconds

        if verbose:
            print(f"Loading tokenizer & embeddings...")
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Use mmap_mode='r' to avoid loading the full 622MB into RAM;
        # accessed rows are paged in on demand from disk/page cache.
        self.embedding_table = np.load(str(embd_path), mmap_mode='r')
        self.embed_dim = self.embedding_table.shape[1]  # 1024

        # Special token IDs (needed before decoder init for n_keep)
        self.ID_IM_START = self._tok1("<|im_start|>")
        self.ID_IM_END = self._tok1("<|im_end|>")
        self.ID_AUDIO_START = self._tok1("<|audio_start|>")
        self.ID_AUDIO_END = self._tok1("<|audio_end|>")
        self.ID_ASR_TEXT = self._tok1("<asr_text>")

        # Build prefix tokens to determine n_keep
        self._prefix_tokens = self._build_prefix_tokens()
        n_keep = len(self._prefix_tokens)

        if verbose:
            print(f"Loading decoder (n_keep={n_keep}, type={decoder_type})...")

        self.decoder_type = decoder_type

        if decoder_type == "matmul":
            # Use MatmulDecoder from rknn-matmul-parallel (no RKLLM conflict).
            # Config is read from config.json in the weights directory.
            # Weights dir: MATMUL_WEIGHTS_DIR env > model_dir/decoder/matmul/ > model_dir
            from .matmul_decoder import MatmulDecoder
            self.decoder = MatmulDecoder(
                model_path=str(model_dir),
                tokenizer=self.tokenizer,
                max_context_len=max_context_len,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                enabled_cpus=enabled_cpus,
                exec_mode=decoder_exec_mode,
                quant_type=os.environ.get("MATMUL_QUANT_TYPE", "int4"),
                callback_fn=decoder_callback,
            )
        else:
            # Use RKLLMDecoder (default)
            from .decoder import RKLLMDecoder
            self.decoder = RKLLMDecoder(
                model_path=str(decoder_path),
                lib_path=str(lib_path),
                max_context_len=max_context_len,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                enabled_cpus=enabled_cpus,
                embed_flash=embed_flash,
                n_keep=n_keep,
                callback_fn=decoder_callback,
            )

        # KV cache reuse disabled — EMBED mode has RoPE position mismatch.
        self._prefix_kv_cached = False

        load_time = time.time() - t_start
        if verbose:
            print(f"\n=== Engine ready in {load_time:.1f}s ===")
            print(f"  Encoder sizes: {self.encoder.available_sizes} "
                  f"({self.encoder.mode})")
            print(f"  Decoder: {decoder_type} ({decoder_exec_mode if decoder_type == 'matmul' else f'{enabled_cpus} CPUs'}), "
                  f"max_new_tokens={max_new_tokens}")
            print(f"  Embedding: {self.embedding_table.shape}")
            print(f"  Prefix KV cache: {len(self._prefix_tokens)} tokens"
                  f" ({'active' if self._prefix_kv_cached else 'inactive'})")

    def _tok1(self, text: str) -> int:
        ids = self.tokenizer.encode(text).ids
        assert len(ids) == 1, f"'{text}' -> {ids}, expected single token"
        return ids[0]

    def _encode_text(self, text: str) -> list:
        return self.tokenizer.encode(text).ids

    def _build_prefix_tokens(self, context: str = "") -> list:
        """Build the fixed prefix token sequence (system prompt + user header)."""
        tk = self._encode_text
        system_text = "You are a helpful assistant. "
        if context:
            system_text += context
        return (
            [self.ID_IM_START] + tk(f"system\n{system_text}") + [self.ID_IM_END]
            + [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]
        )

    def build_embed(self, audio_embd: np.ndarray, prefix_text: str = "",
                    language: str = None, context: str = "",
                    skip_prefix: bool = False) -> tuple:
        """
        Build full embedding for RKLLM: [system | audio | suffix + prefix].
        
        Prompt template:
            <|im_start|>system
            You are a helpful assistant. {context}<|im_end|>
            <|im_start|>user
            <|audio_start|>{AUDIO_EMBED}<|audio_end|>
            数字用0123456789，语音转录：<|im_end|>
            <|im_start|>assistant
            language {Language}<asr_text>{prefix_text}
        
        Args:
            audio_embd: (N, embed_dim) audio embeddings
            prefix_text: Previously recognized text prefix
            language: Language hint (None = auto-detect mode)
            context: Context description
            skip_prefix: If True, omit prefix tokens (use with KV cache)
            
        Returns:
            (full_embed, n_tokens) tuple
        """
        tk = self._encode_text

        if skip_prefix:
            prefix_tokens = []
        else:
            # Rebuild if context changed (rare)
            if context:
                prefix_tokens = self._build_prefix_tokens(context)
            else:
                prefix_tokens = self._prefix_tokens

        assistant_prompt = "assistant\n"
        if language:
            assistant_prompt += f"language {language}"
        else:
            # Auto-detect: let decoder generate "language XXX" itself
            assistant_prompt += "language "

        # Compact suffix saves ~120ms per chunk by reducing token count
        # Full:    "数字用0123456789，语音转录："  (~9 tokens)
        # Compact: "转录："                       (~3 tokens)
        if self.compact_suffix:
            instruction = "转录："
        else:
            instruction = "数字用0123456789，语音转录："

        suffix_tokens = (
            [self.ID_AUDIO_END]
            + tk(instruction)
            + [self.ID_IM_END]
            + [self.ID_IM_START]
            + tk(assistant_prompt)
            + [self.ID_ASR_TEXT]
        )
        if prefix_text:
            suffix_tokens += tk(prefix_text)

        n_pre = len(prefix_tokens)
        n_audio = audio_embd.shape[0]
        n_suf = len(suffix_tokens)
        total = n_pre + n_audio + n_suf

        full_embd = np.zeros((total, self.embed_dim), dtype=np.float32)
        if n_pre > 0:
            full_embd[:n_pre] = self.embedding_table[prefix_tokens]
        full_embd[n_pre:n_pre + n_audio] = audio_embd
        full_embd[n_pre + n_audio:] = self.embedding_table[suffix_tokens]

        return full_embd, total

    def transcribe(self, audio, language: str = "Chinese",
                   context: str = "",
                   chunk_size: float = 30.0,
                   memory_num: int = 2,
                   rollback_tokens: int = 5,
                   max_new_tokens: int = 100,
                   start_second: float = 0.0,
                   duration: float = None,
                   max_chunks: int = None,
                   apply_itn_flag: bool = True) -> dict:
        """
        Transcribe audio file or numpy array.

        Internally creates a StreamSession for consistent sliding-window,
        rollback, and decode logic.

        Args:
            audio: File path (str) or numpy array (float32, 16kHz mono)
            language: Language hint ("Chinese", "English", None for auto)
            context: Context description
            chunk_size: Seconds per chunk (max = encoder capacity)
            memory_num: Chunks in sliding window
            rollback_tokens: Prefix rollback for stability
            max_new_tokens: Max tokens per chunk decode
            start_second: Start offset in seconds
            duration: Duration in seconds (None = full)
            max_chunks: Maximum chunks to process (None = all)
            apply_itn_flag: Apply ITN to final result

        Returns:
            dict: {"text": str, "language": str, "stats": dict}
        """
        # Load audio
        if isinstance(audio, str):
            audio_data = load_audio(audio, start_second=start_second,
                                    duration=duration)
        elif isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
            if start_second > 0:
                audio_data = audio_data[int(start_second * SAMPLE_RATE):]
            if duration:
                audio_data = audio_data[:int(duration * SAMPLE_RATE)]
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        chunk_size = min(chunk_size, self.max_chunk_seconds)
        audio_seconds = len(audio_data) / SAMPLE_RATE
        samples_per = int(chunk_size * SAMPLE_RATE)
        num_chunks = max(1, int(np.ceil(audio_seconds / chunk_size)))
        if max_chunks is not None:
            num_chunks = min(num_chunks, max_chunks)

        if self.verbose:
            print(f"\n[Transcribe] Audio: {audio_seconds:.1f}s, "
                  f"{num_chunks} chunks × {chunk_size}s, "
                  f"memory={memory_num}")

        # Use StreamSession for unified sliding-window logic
        stream = StreamSession(
            engine=self, language=language, context=context,
            chunk_size=chunk_size, memory_num=memory_num,
            unfixed_chunks=0, rollback_tokens=rollback_tokens,
            max_new_tokens=max_new_tokens,
        )

        t_start = time.perf_counter()
        total_len = len(audio_data)
        for i in range(num_chunks):
            s = i * samples_per
            e = min(s + samples_per, total_len)
            stream.feed_audio(audio_data[s:e])

        result = stream.finish(apply_itn_flag=apply_itn_flag)
        wall_ms = (time.perf_counter() - t_start) * 1000

        stats = result["stats"]
        stats["wall_ms"] = wall_ms
        stats["audio_s"] = audio_seconds
        stats["rtf"] = wall_ms / 1000.0 / max(audio_seconds, 0.01)
        stats["chunks"] = num_chunks
        # Backward-compatible aliases
        stats["enc_ms"] = stats["total_enc_ms"]
        stats["llm_ms"] = stats["total_llm_ms"]

        if self.verbose:
            print(f"\n[Total] audio={audio_seconds:.1f}s "
                  f"enc={stats['total_enc_ms']:.0f}ms "
                  f"llm={stats['total_llm_ms']:.0f}ms "
                  f"wall={wall_ms:.0f}ms rtf={stats['rtf']:.3f}")

        return {
            "text": result["text"],
            "language": result["language"],
            "stats": stats,
        }

    def create_stream(self, language: str = "Chinese",
                      context: str = "",
                      chunk_size: float = 4.0,
                      memory_num: int = 2,
                      unfixed_chunks: int = 0,
                      rollback_tokens: int = 2,
                      max_new_tokens: int = 128,
                      on_text: Callable = None,
                      vad = None) -> StreamSession:
        """
        Create a streaming ASR session.
        
        Args:
            language: Language hint
            context: Context string
            chunk_size: Seconds per audio chunk
            memory_num: Sliding window size (min 2)
            unfixed_chunks: First N chunks without prefix text
            rollback_tokens: Prefix rollback tokens
            max_new_tokens: Max tokens per decode
            on_text: Callback called with current full text
            vad: Optional SileroVAD instance for speech gating
            
        Returns:
            StreamSession instance
        """
        return StreamSession(
            engine=self,
            language=language,
            context=context,
            chunk_size=chunk_size,
            memory_num=memory_num,
            unfixed_chunks=unfixed_chunks,
            rollback_tokens=rollback_tokens,
            max_new_tokens=max_new_tokens,
            on_text=on_text,
            vad=vad,
        )

    def close(self):
        """Release all resources."""
        if hasattr(self, 'encoder'):
            self.encoder.release()
        if hasattr(self, 'decoder'):
            self.decoder.release()
        if self.verbose:
            print("[Engine] Released.")

    # ---- File finding helpers ----

    @staticmethod
    def _find_file(directory: Path, pattern: str, desc: str,
                   required: bool = True) -> Optional[Path]:
        import glob
        directory = Path(directory)
        if not directory.exists():
            if required:
                raise FileNotFoundError(
                    f"{desc}: directory not found: {directory}")
            return None

        matches = sorted(directory.glob(pattern))
        if not matches:
            if required:
                raise FileNotFoundError(
                    f"{desc}: no match for '{pattern}' in {directory}")
            return None

        return matches[0]

    @staticmethod
    def _find_tokenizer(model_dir: Path) -> str:
        """Search for tokenizer.json in common locations."""
        candidates = [
            model_dir / "tokenizer.json",
            model_dir / "tokenizer" / "tokenizer.json",
        ]
        import glob
        for p in glob.glob(str(model_dir / "**" / "tokenizer.json"),
                           recursive=True):
            candidates.append(Path(p))

        for c in candidates:
            if c.exists():
                return str(c)

        raise FileNotFoundError(
            f"tokenizer.json not found. Searched in {model_dir} "
            "and subdirectories."
        )
