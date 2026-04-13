"""
Matmul-based Decoder wrapper for Qwen3-ASR.

Replaces RKLLM to avoid RKLLM/RKNN conflict on RK3576.
Uses the matmul_decoder C extension from rknn-matmul-parallel.

The C extension reads model config from config.json in the weights directory.
No hardcoded model parameters here — all config comes from the weights dir.

Performance: ~16ms/token with dual-core NPU parallelism.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class MatmulDecoder:
    """
    Matmul-based decoder using rknn-matmul-parallel C extension.

    Provides the same interface as RKLLMDecoder for seamless replacement.
    Config is loaded from config.json in the weights directory by the C extension.
    """

    def __init__(self,
                 model_path: str,
                 model_dir: str = None,
                 tokenizer=None,
                 max_context_len: int = 4096,
                 max_new_tokens: int = 500,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repeat_penalty: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 enabled_cpus: int = 2,
                 exec_mode: str = "dual_core",
                 quant_type: str = "int4",
                 callback_fn: Callable = None):
        """
        Initialize matmul decoder.

        Args:
            model_path: Path to ASR model root directory. Matmul weights are
                        looked up in model_path/decoder/matmul/ by default,
                        or override via MATMUL_WEIGHTS_DIR env var.
            model_dir: Alternative path (deprecated, use model_path)
            tokenizer: Tokenizer instance for decoding
            max_context_len: Maximum KV cache size
            max_new_tokens: Maximum tokens to generate per call
            top_k: Top-K sampling (1 = greedy)
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            repeat_penalty: Repetition penalty
            enabled_cpus: Number of CPU cores (unused, kept for interface compat)
            exec_mode: "single_core" or "dual_core"
            quant_type: "fp16", "int4", "int8", or "int4_g128"
            callback_fn: Optional callback(text, is_finish) for streaming
        """
        # Import C extension from rknn-matmul-parallel
        try:
            import matmul_decoder as md
            self._md = md
        except ImportError as e:
            raise ImportError(
                "matmul_decoder C extension not found. "
                "Build it from ~/project/rknn-matmul-parallel: make python\n"
                "Then set PYTHONPATH to include that directory, or mount it "
                "into the Docker container."
            ) from e

        # Resolve matmul weights directory
        if model_dir:
            model_path = model_dir
        base_path = Path(model_path)

        # Priority: MATMUL_WEIGHTS_DIR env > model_path/decoder/matmul/ > model_path
        env_weights = os.environ.get("MATMUL_WEIGHTS_DIR")
        if env_weights:
            self.weights_dir = Path(env_weights)
        elif (base_path / "decoder" / "matmul").exists():
            self.weights_dir = base_path / "decoder" / "matmul"
        else:
            self.weights_dir = base_path

        # Store params
        self.max_context_len = max_context_len
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.callback_fn = callback_fn
        self._seq_len = 0
        self._tokenizer = tokenizer
        self._eos_token_id = 151645  # <|im_end|> for Qwen

        # Create decoder — config is read from weights_dir/config.json by C extension
        t0 = time.time()
        weights_ready = self._check_weights_ready()

        if weights_ready:
            self._decoder = md.MatmulDecoder(
                model_dir=str(self.weights_dir),
                max_seq_len=max_context_len,
                quant_type=quant_type,
                exec_mode=exec_mode,
            )
            load_time = time.time() - t0
            # Read config back from the C extension for logging
            cfg = self._decoder.config
            logger.info("MatmulDecoder loaded in %.1fs (mode=%s, %d layers, hidden=%d)",
                        load_time, exec_mode, cfg.get("num_layers", 0), cfg.get("hidden_dim", 0))
            print(f"[MatmulDecoder] Loaded from {self.weights_dir}. "
                  f"mode={exec_mode} quant={quant_type} "
                  f"hidden={cfg.get('hidden_dim')} layers={cfg.get('num_layers')}")
        else:
            self._decoder = None
            logger.warning(
                "MatmulDecoder weights not found at %s. "
                "Required: config.json, embeddings.npy, layers/ directory.\n"
                "Export with: python scripts/export_qwen3_asr_weights.py --output %s",
                self.weights_dir, self.weights_dir,
            )
            print(f"[MatmulDecoder] Weights not found at {self.weights_dir}")

    def _check_weights_ready(self) -> bool:
        """Check if matmul weights are available."""
        required = [
            self.weights_dir / "config.json",
            self.weights_dir / "embeddings.npy",
        ]
        layers_dir = self.weights_dir / "layers"
        return all(f.exists() for f in required) and layers_dir.exists()

    @property
    def seq_len(self) -> int:
        """Current sequence length in KV cache."""
        return self._seq_len

    def clear_kv_cache(self):
        """Clear KV cache for new sequence."""
        if self._decoder:
            self._decoder.clear_kv_cache()
        self._seq_len = 0

    def run_embed(self,
                  embed_array: np.ndarray,
                  n_tokens: int,
                  keep_history: int = 0,
                  keep_prefix: bool = False) -> dict:
        """
        Run decoder with embedding input.

        This is the main interface matching RKLLMDecoder.run_embed().

        Args:
            embed_array: (n_tokens, hidden_dim) float32 embedding
            n_tokens: Number of tokens
            keep_history: 0 = clear KV cache, 1 = keep history
            keep_prefix: Not used (KV cache always cleared per call for ASR)

        Returns:
            dict with keys: text, perf, n_tokens_generated
        """
        if self._decoder is None:
            raise RuntimeError(
                "MatmulDecoder not initialized — weights not found. "
                "Export weights first or set MATMUL_WEIGHTS_DIR."
            )

        if keep_history == 0:
            self.clear_kv_cache()

        t0 = time.time()
        generated_tokens = []

        # Prefill: feed all embedding tokens to build KV cache.
        # All but the last token use prefill() which skips final_norm + lm_head,
        # saving ~35ms per token. Only the last token computes logits.
        t_prefill = time.time()
        for i in range(n_tokens - 1):
            self._decoder.prefill(token_id=-1, embedding=embed_array[i])
        first_token = self._decoder.step_get_token(
            token_id=-1, embedding=embed_array[n_tokens - 1]
        )
        prefill_ms = (time.time() - t_prefill) * 1000

        # Generate tokens autoregressively
        # Start from the token predicted during prefill's last step
        t_gen = time.time()
        prev_token = first_token
        generated_tokens.append(prev_token)
        for _ in range(self.max_new_tokens - 1):
            token = self._decoder.step_get_token(token_id=prev_token)
            generated_tokens.append(token)
            prev_token = token

            if token == self._eos_token_id:
                break

            if self.callback_fn and self._tokenizer:
                text = self._tokenizer.decode([token])
                self.callback_fn(text, False)

        gen_time = time.time() - t0
        gen_only_ms = (time.time() - t_gen) * 1000

        if self.callback_fn:
            self.callback_fn("", True)

        # Decode tokens to text
        if self._tokenizer:
            text = self._tokenizer.decode(generated_tokens)
            for tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                text = text.replace(tag, "")
        else:
            text = self._decode_tokens_fallback(generated_tokens)

        self._seq_len = n_tokens + len(generated_tokens)

        return {
            "text": text,
            "perf": {
                "prefill_time_ms": prefill_ms,
                "prefill_tokens": n_tokens,
                "generate_time_ms": gen_only_ms,
                "generate_tokens": len(generated_tokens),
                "ms_per_prefill_token": prefill_ms / max(n_tokens, 1),
                "ms_per_gen_token": gen_only_ms / max(len(generated_tokens), 1),
            },
            "n_tokens_generated": len(generated_tokens),
            "ret_code": 0,
            "aborted": False,
        }

    def _decode_tokens_fallback(self, token_ids: list[int]) -> str:
        """Fallback decoder when tokenizer not available."""
        return "".join(chr(t) if 32 <= t < 127 else "" for t in token_ids)

    def release(self):
        """Release resources."""
        if hasattr(self, '_decoder') and self._decoder is not None:
            del self._decoder
            self._decoder = None

    def __del__(self):
        self.release()


class MatmulDecoderWrapper:
    """
    Wrapper that provides RKLLMDecoder-compatible interface.

    Used to replace RKLLMDecoder in Qwen3ASREngine without code changes.
    """

    def __init__(self, **kwargs):
        self._impl = MatmulDecoder(**kwargs)

    def run_embed(self, *args, **kwargs):
        return self._impl.run_embed(*args, **kwargs)

    def clear_kv_cache(self):
        return self._impl.clear_kv_cache()

    def release(self):
        return self._impl.release()

    @property
    def seq_len(self):
        return self._impl.seq_len


def create_decoder(model_path: str,
                   use_matmul: bool = True,
                   exec_mode: str = "dual_core",
                   **kwargs) -> MatmulDecoderWrapper:
    """
    Create decoder instance.

    Args:
        model_path: Path to model weights
        use_matmul: If True, use MatmulDecoder; else use RKLLMDecoder
        exec_mode: "single_core" or "dual_core" (only for matmul)
        **kwargs: Additional arguments passed to decoder

    Returns:
        Decoder instance with unified interface
    """
    if use_matmul:
        kwargs["exec_mode"] = exec_mode
        return MatmulDecoderWrapper(model_path=model_path, **kwargs)
    else:
        from .decoder import RKLLMDecoder
        return RKLLMDecoder(model_path=model_path, **kwargs)
