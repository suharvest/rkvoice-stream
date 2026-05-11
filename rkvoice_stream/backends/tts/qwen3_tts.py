"""Qwen3-TTS inference service for RK3576.

Pipeline:
  1. Tokenize text (transformers tokenizer)
  2. text_project (RKNN NPU) -> text embeddings [T, 1024]
  3. Build prefill embeddings (numpy element-wise add)
  4. talker prefill (RKLLM, mode=1) -> hidden states
  5. logits = hidden @ codec_head (numpy matmul)
  6. AR loop:
     a. sample primary token from logits
     b. code_predictor (RKNN NPU) -> residual codes
     c. build next embedding (numpy: sum 16 codebook lookups + text embed)
     d. talker decode (RKLLM, mode=1, keep_history=1) -> hidden
     e. logits = hidden @ codec_head
  7. vocoder (RKNN NPU, decoder_ctx25_int8) -> audio waveform
  8. Return WAV
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Iterator, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

HIDDEN_SIZE = 1024
NUM_CODE_GROUPS = 16
TALKER_VOCAB_SIZE = 3072
CODE_PREDICTOR_VOCAB_SIZE = 2048

# Vocoder params - tokenizer12hz_decode_stream outputs 1920 samples per frame
# at 24kHz sample rate with 12.5 Hz frame rate
SAMPLE_RATE = 24000  # 24kHz output
SAMPLES_PER_FRAME = 1920  # 24000 / 12.5 = 1920 samples per codec frame

# Special token IDs (from Qwen3-TTS config)
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671

CODEC_BOS_ID = 2149
CODEC_EOS_TOKEN_ID = 2150
CODEC_PAD_ID = 2148
CODEC_NOTHINK_ID = 2155
CODEC_THINK_ID = 2154  # Added for official prefill compatibility
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Language IDs for codec prefix (from official Qwen3-TTS)
LANG_ID_CHINESE = 2055
LANG_ID_ENGLISH = 2050
LANG_ID_JAPANESE = 2058
LANG_ID_KOREAN = 2064

# Vocoder params
VOCODER_CTX_FRAMES = 25  # context frames for streaming decoder

# EOS termination: W4A16 quantization may suppress EOS logit, so we add a
# progressive bias after an estimated speech duration.  Frames per char is
# ~6-8 for Chinese, ~4-5 for English.  The bias ramps linearly after the
# estimated duration so silence/repetition is cut off.
FRAMES_PER_CHAR = 8       # conservative estimate
EOS_BIAS_RAMP_FRAMES = 50 # ramp from 0 to EOS_BIAS_MAX over this many frames
EOS_BIAS_MAX = 15.0       # enough to dominate most logit ranges


class TTSService:
    """Full Qwen3-TTS pipeline on RK3576."""

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._ready = False

        self._tokenizer = None
        self._talker = None
        self._text_project = None
        self._codec_embed = None
        self._code_predictor = None
        self._code_predictor_embed = None
        self._vocoder = None
        self._vocoder_name = None  # RKNN model name for reload
        self._codec_head_weight = None
        self._codebook_embeds = None
        self._cp_engine = None  # C engine for code_predictor (optional)
        self._codec_embed_table = None  # numpy lookup table (optional, replaces RKNN)
        self._text_project_table = None  # numpy lookup table (optional, replaces RKNN)

    def load(self):
        """Load all models. Call once at startup."""
        t0 = time.perf_counter()

        self._load_tokenizer()
        self._load_numpy_tables()  # text_project + codec_embed (before RKNN, to skip unneeded models)
        self._load_rknn_models()
        self._load_rkllm_talker()
        self._load_numpy_weights()
        self._load_cp_engine()

        elapsed = time.perf_counter() - t0
        logger.info("All models loaded in %.1fs", elapsed)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_sample_rate(self) -> int:
        return SAMPLE_RATE

    # ── Model Loading ────────────────────────────────────────────

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok_path = os.path.join(self._model_dir, "tokenizer")
        logger.info("Loading tokenizer from %s", tok_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True
        )

    def _load_rknn_models(self):
        from rknnlite.api import RKNNLite

        def load_rknn(name: str, core_mask=None) -> RKNNLite:
            path = os.path.join(self._model_dir, f"{name}.rknn")
            logger.info("Loading RKNN: %s", path)
            rknn = RKNNLite(verbose=False)
            ret = rknn.load_rknn(path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN {path}: ret={ret}")
            if core_mask is None:
                core_mask = RKNNLite.NPU_CORE_AUTO
            ret = rknn.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN runtime {name}: ret={ret}")
            return rknn

        # Skip RKNN models when numpy/C engine replacements exist
        if self._text_project_table is None:
            self._text_project = load_rknn("text_project")
        else:
            logger.info("Skipping text_project RKNN (numpy table loaded, saving ~600MB)")

        if self._codec_embed_table is None:
            self._codec_embed = load_rknn("codec_embed")
        else:
            logger.info("Skipping codec_embed RKNN (numpy table loaded)")

        # Always load code_predictor RKNN as fallback (C engine replaces at runtime if loaded)
        self._code_predictor = load_rknn("code_predictor")
        self._code_predictor_embed = load_rknn("code_predictor_embed")

        # Vocoder selection: prefer noembed type (decoder_ctx25_fp16/int8)
        # tokenizer12hz_decode_stream model has issues (outputs zeros)
        # decoder_ctx25_fp16 works correctly even after RKLLM usage
        fp16_path = os.path.join(self._model_dir, "decoder_ctx25_fp16.rknn")
        int8_path = os.path.join(self._model_dir, "decoder_ctx25_int8.rknn")
        stream_path = os.path.join(self._model_dir, "tokenizer12hz_decode_stream.rknn")
        if os.path.exists(fp16_path):
            self._vocoder_name = "decoder_ctx25_fp16"
            self._vocoder = load_rknn(self._vocoder_name)
            self._vocoder_type = "noembed"  # input: [1, 512, 50] float32
            logger.info("Using FP16 vocoder (decoder_ctx25_fp16, ~1.3s/chunk)")
        elif os.path.exists(int8_path):
            self._vocoder_name = "decoder_ctx25_int8"
            self._vocoder = load_rknn(self._vocoder_name)
            self._vocoder_type = "noembed"  # input: [1, 512, 50] float32
            logger.info("Using INT8 vocoder (decoder_ctx25_int8, ~620ms/chunk)")
        elif os.path.exists(stream_path):
            self._vocoder_name = "tokenizer12hz_decode_stream"
            self._vocoder = load_rknn(self._vocoder_name)
            self._vocoder_type = "stream"  # input: [1, 75, 16] int64
            logger.info("Using stream vocoder (tokenizer12hz_decode_stream)")
        else:
            raise FileNotFoundError("No vocoder model found")

    def _reload_vocoder(self) -> None:
        """Release and reload the vocoder RKNN context.

        Workaround for RKNPU driver bug (v0.9.8): the Conv layer in decoder
        models (decoder.4/block.2/conv2/conv/Conv_ConvAdd) causes NPU job
        timeout after the first successful inference.  Reloading the RKNN
        context before each inference gives it a fresh context so each call
        succeeds (the bug only appears on the *second* call per context).
        """
        from rknnlite.api import RKNNLite

        if self._vocoder is not None:
            try:
                self._vocoder.release()
            except Exception:
                pass
            self._vocoder = None

        path = os.path.join(self._model_dir, f"{self._vocoder_name}.rknn")
        logger.debug("Reloading vocoder context: %s", path)
        rknn = RKNNLite(verbose=False)
        ret = rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"Failed to reload vocoder {path}: ret={ret}")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init reloaded vocoder runtime: ret={ret}")
        self._vocoder = rknn

    def _load_rkllm_talker(self):
        from rkvoice_stream.runtime.rkllm_wrapper import RKLLMTalker

        rkllm_path = os.path.join(self._model_dir, "talker_fullvocab_fixed_w4a16_rk3576.rkllm")
        self._talker = RKLLMTalker(
            model_path=rkllm_path,
            rkllm_lib="/usr/lib/librkllmrt.so",
            rknn_lib="librknnrt.so",
            max_context_len=512,
            max_new_tokens=1,  # Step-by-step: 1 token per call
        )

    def _load_numpy_weights(self):
        logger.info("Loading numpy weights...")

        # codec_head: [3072, 1024] for logits = hidden @ codec_head.T
        ch_path = os.path.join(self._model_dir, "codec_head_weight.npy")
        self._codec_head_weight = np.load(ch_path).astype(np.float32)
        logger.info("codec_head: %s", self._codec_head_weight.shape)

        # codebook embeddings: 16 codebooks of [2048, 256]
        cb_dir = os.path.join(self._model_dir, "codebook_embeds")
        self._codebook_embeds = []
        for i in range(NUM_CODE_GROUPS):
            cb = np.load(os.path.join(cb_dir, f"codebook_{i}.npy")).astype(np.float32)
            self._codebook_embeds.append(cb)
        logger.info("Loaded %d codebook embeddings: %s each", len(self._codebook_embeds), self._codebook_embeds[0].shape)

        # output_proj weights for code_predictor logit computation
        self._output_proj_first = np.load(os.path.join(cb_dir, "output_proj_first_weight.npy")).astype(np.float32)
        self._output_proj_rest = np.load(os.path.join(cb_dir, "output_proj_rest_weight.npy")).astype(np.float32)

    def _load_numpy_tables(self):
        """Load embedding tables as numpy (mmap) to replace heavy RKNN models."""
        # text_project: [vocab_size, 1024] — replaces 606MB RKNN
        tp_path = os.path.join(self._model_dir, "text_project_table.npy")
        if os.path.exists(tp_path):
            self._text_project_table = np.load(tp_path, mmap_mode='r')
            logger.info("text_project table loaded: %s (mmap, replaces 606MB RKNN)", self._text_project_table.shape)
        else:
            logger.info("text_project_table.npy not found, will use RKNN")

        # codec_embed: [vocab_size, 1024] — replaces 6MB RKNN
        ce_path = os.path.join(self._model_dir, "codec_embed_table.npy")
        if os.path.exists(ce_path):
            self._codec_embed_table = np.load(ce_path, mmap_mode='r')
            logger.info("codec_embed table loaded: %s (mmap)", self._codec_embed_table.shape)
        else:
            logger.info("codec_embed_table.npy not found, will use RKNN")

    def _load_cp_engine(self):
        """Try to load C engine for code_predictor (faster than 15x RKNN calls)."""
        try:
            import sys
            engine_dir = os.path.join(os.path.dirname(__file__), "..", "engine")
            lib_path = os.path.join(os.path.dirname(__file__), "..", "lib", "libcp_engine.so")
            if not os.path.exists(lib_path):
                lib_path = os.path.join(engine_dir, "libcp_engine.so")
            if not os.path.exists(lib_path):
                logger.info("C engine not found, using RKNN code_predictor fallback")
                return

            sys.path.insert(0, engine_dir)
            from cp_engine_wrapper import CodePredictorEngine

            weight_dir = os.path.join(self._model_dir, "cp_weights")
            if not os.path.isdir(weight_dir):
                logger.info("cp_weights dir not found at %s, skipping C engine", weight_dir)
                return

            self._cp_engine = CodePredictorEngine(weight_dir, num_npu_cores=2, lib_path=lib_path)
            logger.info("C engine loaded (W8A16 matmul API)")
        except Exception as e:
            logger.warning("C engine load failed, using RKNN fallback: %s", e)
            self._cp_engine = None

    # ── RKNN Helpers ─────────────────────────────────────────────

    def _run_text_project(self, token_ids: list[int]) -> np.ndarray:
        """Lookup text embeddings. Uses numpy table if available, else RKNN.
        Returns [N, 1024] where N = len(token_ids).
        """
        if self._text_project_table is not None:
            return np.array(self._text_project_table[token_ids], dtype=np.float32)
        n = len(token_ids)
        padded = np.zeros((1, 128), dtype=np.int64)
        use_n = min(n, 128)
        padded[0, :use_n] = token_ids[:use_n]
        outputs = self._text_project.inference(inputs=[padded])
        result = np.array(outputs[0])  # [1, 128, 1024]
        return result[0, :use_n]  # [N, 1024]

    def _run_codec_embed(self, codec_ids: list[int]) -> np.ndarray:
        """Lookup codec embeddings. Uses numpy table if available, else RKNN.
        Returns [N, 1024].
        """
        if self._codec_embed_table is not None:
            return self._codec_embed_table[codec_ids]  # [N, 1024]
        results = []
        for cid in codec_ids:
            inp = np.array([[cid]], dtype=np.int64)
            out = self._codec_embed.inference(inputs=[inp])
            results.append(np.array(out[0])[0, 0])  # [1024]
        return np.stack(results)  # [N, 1024]

    def _run_code_predictor(self, context: np.ndarray) -> np.ndarray:
        """Run code_predictor RKNN.
        Args:
            context: [1, 2, 1024] float32 (fixed 2-token input for static RKNN model)
        Returns: logits [1, 1, 2048] float32
        """
        outputs = self._code_predictor.inference(inputs=[context])
        return np.array(outputs[0])  # [1, 1, 2048]

    def _run_code_predictor_embed(self, code_id: int, gen_step: int) -> np.ndarray:
        """Run code_predictor_embed RKNN.
        Returns: [1024] float32 embedding for residual code.
        """
        inp_id = np.array([[code_id]], dtype=np.int64)
        gs = np.array([gen_step], dtype=np.int64)
        outputs = self._code_predictor_embed.inference(inputs=[inp_id, gs])
        return np.array(outputs[0])[0, 0]  # [1024]

    def _run_vocoder_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Run vocoder RKNN (decoder_ctx25_int8) on pre-computed embeddings.
        Args:
            embeddings: [1, 512, T] float32
        Returns: audio samples float32
        """
        outputs = self._vocoder.inference(inputs=[embeddings])
        return np.array(outputs[0]).flatten()

    # ── Sampling ─────────────────────────────────────────────────

    @staticmethod
    def _apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float = 1.05) -> np.ndarray:
        """Apply repetition penalty to logits based on previously generated tokens."""
        if not generated_ids or penalty == 1.0:
            return logits
        logits = logits.copy()
        unique_ids = set(generated_ids)
        for token_id in unique_ids:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    @staticmethod
    def _sample_top_k(logits: np.ndarray, top_k: int = 5, temperature: float = 0.8, eos_id: int | None = None) -> int:
        """Sample from logits with top-k and temperature.
        If eos_id is given, always include it in the candidate set so EOS can be sampled."""
        if temperature <= 0 or top_k <= 1:
            return int(np.argmax(logits))

        logits = logits / temperature
        # Top-k filtering
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]

        # Ensure EOS is always a candidate when requested
        if eos_id is not None and eos_id not in top_k_idx:
            # Replace the lowest-scoring candidate with EOS
            worst = np.argmin(logits[top_k_idx])
            top_k_idx[worst] = eos_id

        top_k_logits = logits[top_k_idx]
        # Softmax
        top_k_logits -= top_k_logits.max()
        probs = np.exp(top_k_logits)
        probs /= probs.sum()
        chosen = np.random.choice(top_k_idx, p=probs)
        return int(chosen)

    # ── Main Synthesis ───────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        temperature: float = 0.8,
        top_k: int = 5,
        max_new_tokens: int = 300,
        language: str = "chinese",
    ) -> tuple[bytes, dict]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_id: Speaker ID (not used in current implementation)
            speed: Speed multiplier (1.0 = normal)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum frames to generate
            language: Language for lang_id ("chinese", "english", "japanese", "korean")

        Returns: (wav_bytes, metadata_dict)
        """
        t_start = time.perf_counter()

        # Step 0: Reload talker if destroyed by previous vocoder step
        if self._talker is None:
            self._load_rkllm_talker()

        # Step 1: Tokenize
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._tokenizer.encode(formatted_text)
        logger.info("Tokenized: %d tokens", len(input_ids))

        # Step 2: Build prefill embeddings (official format: all text in prefill)
        prefill_embeds, tts_pad_vec = self._build_prefill(input_ids, language=language)
        t_prefill_built = time.perf_counter()
        logger.info("Prefill embeddings: %s (%.0fms)", prefill_embeds.shape,
                     (t_prefill_built - t_start) * 1000)

        # Step 3: Talker prefill
        # clear_kv_cache wipes everything; keep_history=1 then says "keep the result
        # of this call in cache" — without it, RKLLM v1.2.3 clears KV after the run
        # and decode steps see empty context, generating training-data priors.
        self._talker.clear_kv_cache()
        result = self._talker.run_embed(
            prefill_embeds,
            mode=1,  # GET_LAST_HIDDEN_LAYER -> returns hidden states
            keep_history=1,
        )
        hidden = result["hidden"]  # [n_prefill, 1024]
        last_hidden = hidden[-1:]  # [1, 1024]

        # Step 4: Compute initial logits
        logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]
        t_prefill_done = time.perf_counter()
        logger.info("Prefill done: %.0fms", (t_prefill_done - t_prefill_built) * 1000)

        # Step 5: AR generate loop
        all_codes = []
        min_new_tokens = 2

        # EOS bias: estimate expected duration and ramp bias after that
        text_chars = len(text.strip())
        est_frames = max(text_chars * FRAMES_PER_CHAR, 10)
        # Dynamic cap: prevent runaway generation (3x estimated + minimum 30)
        max_new_tokens = min(max_new_tokens, max(text_chars * FRAMES_PER_CHAR * 3, 30))
        logger.info("EOS bias: est_frames=%d, max_tokens=%d for %d chars",
                     est_frames, max_new_tokens, text_chars)

        # Profiling accumulators (first 10 steps only to avoid log spam)
        _prof_embed = 0.0
        _prof_cp = 0.0
        _prof_talker = 0.0
        _prof_head = 0.0
        _prof_other = 0.0
        _prof_n = 0

        # Suppress control tokens [2048, 3072) except EOS — only codec tokens [0, 2048) and EOS are valid
        suppress_mask = np.ones(TALKER_VOCAB_SIZE, dtype=bool)
        suppress_mask[:CODE_PREDICTOR_VOCAB_SIZE] = False  # allow [0, 2048)
        suppress_mask[CODEC_EOS_TOKEN_ID] = False          # allow EOS
        suppress_indices = np.where(suppress_mask)[0]

        generated_primary_codes = []  # for repetition penalty

        for step in range(max_new_tokens):
            t_step = time.perf_counter()

            # Suppress control tokens (always)
            logits[suppress_indices] = -float("inf")

            # Apply repetition penalty (official: 1.05 over full sequence)
            logits = self._apply_repetition_penalty(logits, generated_primary_codes, penalty=1.05)

            # 5a: Sample primary code
            if step < min_new_tokens:
                logits[CODEC_EOS_TOKEN_ID] = -float("inf")

            # Apply progressive EOS bias after estimated duration
            if step >= est_frames:
                ramp = min((step - est_frames) / EOS_BIAS_RAMP_FRAMES, 1.0)
                eos_bias = ramp * EOS_BIAS_MAX
                logits[CODEC_EOS_TOKEN_ID] += eos_bias

            # Debug: log EOS logit rank periodically
            if step % 25 == 0 or step < 5:
                eos_logit = logits[CODEC_EOS_TOKEN_ID]
                argmax_id = int(np.argmax(logits))
                argmax_val = logits[argmax_id]
                rank = int((logits > eos_logit).sum())
                logger.info("Step %d: EOS logit=%.3f (rank %d/%d), argmax=%d (%.3f), bias_active=%s",
                            step, eos_logit, rank, len(logits), argmax_id, argmax_val,
                            step >= est_frames)

            # Greedy EOS check: if EOS is the argmax, stop immediately
            if step >= min_new_tokens and np.argmax(logits) == CODEC_EOS_TOKEN_ID:
                logger.info("EOS at step %d (greedy)", step)
                break

            primary_code = self._sample_top_k(
                logits, top_k=top_k, temperature=temperature,
                eos_id=CODEC_EOS_TOKEN_ID if step >= min_new_tokens else None,
            )

            if primary_code == CODEC_EOS_TOKEN_ID:
                logger.info("EOS at step %d", step)
                break

            generated_primary_codes.append(primary_code)

            # 5b: Get primary embedding via codec_embed RKNN
            t0 = time.perf_counter()
            primary_embed = self._run_codec_embed([primary_code])  # [1, 1024]
            t_embed = time.perf_counter() - t0

            # 5c: Residual codes via code_predictor
            frame_codes = [primary_code]

            t0 = time.perf_counter()
            if self._cp_engine is not None:
                n_residual = NUM_CODE_GROUPS - 1
                codes, codec_sum = self._cp_engine.run(last_hidden[0], primary_embed[0], num_steps=n_residual)
                frame_codes.extend(int(c) for c in codes)
            else:
                codec_sum = primary_embed[0].copy()
                latest_embed = primary_embed[0]
                for j in range(NUM_CODE_GROUPS - 1):
                    cp_input = np.stack([last_hidden[0], latest_embed])[np.newaxis, :, :]
                    cp_logits = self._run_code_predictor(cp_input)
                    cp_logits_last = cp_logits[0, 0]
                    res_code = self._sample_top_k(
                        cp_logits_last[:CODE_PREDICTOR_VOCAB_SIZE],
                        top_k=1, temperature=0.0
                    )
                    frame_codes.append(res_code)
                    res_embed = self._run_code_predictor_embed(res_code, j)
                    latest_embed = res_embed
                    codec_sum += res_embed
            t_cp = time.perf_counter() - t0

            all_codes.append(frame_codes)

            # 5d: Build next talker input (official format: tts_pad + codec_sum)
            # Note: All text is already in prefill, so decode steps only use tts_pad + codec_sum
            next_embed = (tts_pad_vec + codec_sum)[np.newaxis, :]  # [1, 1024]

            # 5e: Talker decode step (keep_history=1 = append to KV cache from prefill)
            t0 = time.perf_counter()
            result = self._talker.run_embed(
                next_embed,
                mode=1,  # GET_LAST_HIDDEN_LAYER -> returns hidden states
                keep_history=1,
            )
            t_talker = time.perf_counter() - t0

            t0 = time.perf_counter()
            last_hidden = result["hidden"][-1:]  # [1, 1024]
            logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]
            t_head = time.perf_counter() - t0

            t_total = time.perf_counter() - t_step
            t_other = t_total - t_embed - t_cp - t_talker - t_head

            # Accumulate for summary
            if step < 10:
                _prof_embed += t_embed
                _prof_cp += t_cp
                _prof_talker += t_talker
                _prof_head += t_head
                _prof_other += t_other
                _prof_n += 1

            if step == 9:
                n = _prof_n
                logger.info(
                    "AR profile (avg of %d steps): embed=%.1fms cp=%.1fms talker=%.1fms head=%.1fms other=%.1fms total=%.1fms",
                    n, _prof_embed/n*1000, _prof_cp/n*1000, _prof_talker/n*1000,
                    _prof_head/n*1000, _prof_other/n*1000,
                    (_prof_embed+_prof_cp+_prof_talker+_prof_head+_prof_other)/n*1000,
                )

            if (step + 1) % 50 == 0:
                logger.info("Generated %d frames", step + 1)

        t_ar_done = time.perf_counter()
        n_frames = len(all_codes)
        ar_time = t_ar_done - t_prefill_done
        logger.info("AR loop: %d frames in %.1fs (%.1f frames/s)",
                     n_frames, ar_time, n_frames / ar_time if ar_time > 0 else 0)

        if n_frames == 0:
            # Return silence
            wav_bytes = self._make_wav(np.zeros(SAMPLE_RATE, dtype=np.float32))
            return wav_bytes, {"duration": 1.0, "inference_time": 0, "rtf": 0}

        # Step 5.5: Not releasing RKLLM — NPU lock at backend level already
        # serializes TTS and ASR. The vocoder (RKNN) and talker (RKLLM) share
        # NPU cores safely as long as they don't run concurrently.

        # Step 6: Vocoder (codes -> embeddings -> audio)
        codes_array = np.array(all_codes, dtype=np.int64)  # [T, NUM_CODE_GROUPS]
        logger.info("Starting vocoder: codes_array shape=%s dtype=%s", codes_array.shape, codes_array.dtype)
        try:
            audio = self._decode_audio(codes_array)
        except Exception as exc:
            logger.error("Vocoder FAILED: %s: %s", type(exc).__name__, exc, exc_info=True)
            raise
        t_vocoder_done = time.perf_counter()
        logger.info("Vocoder: %.1fs", t_vocoder_done - t_ar_done)

        # Apply speed adjustment
        if speed != 1.0 and speed > 0:
            # Simple speed via resampling
            n_orig = len(audio)
            n_new = int(n_orig / speed)
            indices = np.linspace(0, n_orig - 1, n_new)
            audio = np.interp(indices, np.arange(n_orig), audio).astype(np.float32)

        # Step 7: Make WAV
        duration = len(audio) / SAMPLE_RATE
        total_time = time.perf_counter() - t_start
        rtf = total_time / duration if duration > 0 else 0

        wav_bytes = self._make_wav(audio)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(total_time, 3),
            "rtf": round(rtf, 3),
            "frames": n_frames,
            "ar_time": round(ar_time, 3),
        }
        logger.info("TTS done: %.1fs audio in %.1fs (RTF=%.2f)", duration, total_time, rtf)
        return wav_bytes, meta

    # ── Prefill Construction ─────────────────────────────────────

    def _build_prefill(self, input_ids: list[int], language: str = "chinese"):
        """Build prefill embeddings following official Qwen3-TTS layout.

        Official prefill structure (from tts_dump_reference.py):
          1. Role prefix: [IM_START, assistant, newline] → 3 positions
          2. Prefix block: [tts_pad×4, tts_bos] + [think, think_bos, lang_id, think_eos, pad] → 5 positions
          3. Total prefix: 8 positions
          4. Text block: [text_tokens..., tts_eos] + [codec_pad×(N+1)] → N+1 positions
          5. Final: [tts_pad + codec_bos] → 1 position
          6. Total prefill: 9 + N positions

        Args:
            input_ids: Full formatted token IDs from tokenizer
            language: Language for lang_id ("chinese", "english", "japanese", "korean")

        Returns: (prefill_embeds [9+N, 1024], tts_pad_vec ndarray)
        """
        # Get language ID
        lang_id = {
            "chinese": LANG_ID_CHINESE,
            "english": LANG_ID_ENGLISH,
            "japanese": LANG_ID_JAPANESE,
            "korean": LANG_ID_KOREAN,
        }.get(language.lower(), LANG_ID_CHINESE)

        # Special embeddings via text_project
        special_ids = [TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]
        special_embed = self._run_text_project(special_ids)  # [3, 1024]
        tts_bos_embed = special_embed[0]  # [1024]
        tts_eos_embed = special_embed[1]  # [1024]
        tts_pad_embed = special_embed[2]  # [1024]

        # Role embeddings (first 3 tokens: <|im_start|> assistant \n)
        role_ids = input_ids[:3]
        role_embed = self._run_text_project(role_ids)  # [3, 1024]

        # Codec prefix embeddings (6 tokens for official format)
        # [think, think_bos, lang_id, think_eos, pad, bos]
        codec_prefix_ids = [
            CODEC_THINK_ID, CODEC_THINK_BOS_ID, lang_id,
            CODEC_THINK_EOS_ID, CODEC_PAD_ID, CODEC_BOS_ID,
        ]
        codec_prefix_embed = self._run_codec_embed(codec_prefix_ids)  # [6, 1024]

        # Body text: tokens [3:-5] (between role and trailing special tokens)
        text_start = 3
        text_end = len(input_ids) - 5
        body_text_ids = input_ids[text_start:text_end] if text_end > text_start else []

        # Build prefix block (5 positions): [tts_pad×4, tts_bos] + codec_prefix[:-1]
        # Element-wise addition: tts_part + codec_prefix[:5]
        prefix = np.zeros((5, HIDDEN_SIZE), dtype=np.float32)
        prefix[:4] = tts_pad_embed  # tts_pad×4
        prefix[4] = tts_bos_embed   # tts_bos
        prefix = prefix + codec_prefix_embed[:5]  # Add [think, think_bos, lang_id, think_eos, pad]

        # Combine role + prefix → 8 positions
        embed_prefix = np.concatenate([role_embed, prefix], axis=0)  # [8, 1024]

        # Text block: [text_tokens..., tts_eos] + [codec_pad×(N+1)]
        if body_text_ids:
            text_embed = self._run_text_project(body_text_ids)  # [N, 1024]
            text_block = np.concatenate([text_embed, tts_eos_embed[np.newaxis, :]], axis=0)  # [N+1, 1024]
        else:
            text_block = tts_eos_embed[np.newaxis, :]  # [1, 1024]

        # Add codec_pad to text_block
        N_text = text_block.shape[0]
        codec_pad_block = np.tile(codec_prefix_embed[4], (N_text, 1))  # codec_pad = index 4
        text_block = text_block + codec_pad_block  # Element-wise add

        # Final token: tts_pad + codec_bos
        final = tts_pad_embed + codec_prefix_embed[5]  # codec_bos = index 5

        # Combine all: embed_prefix [8] + text_block [N+1] + final [1] = 9+N positions
        prefill = np.concatenate([embed_prefix, text_block, final[np.newaxis, :]], axis=0)

        logger.info("Prefill built: %d positions (8 prefix + %d text + 1 final)",
                    prefill.shape[0], N_text)

        return prefill, tts_pad_embed

    # ── Audio Decode ─────────────────────────────────────────────

    def _codes_to_embeddings(self, codes: np.ndarray) -> np.ndarray:
        """Convert discrete codec codes to continuous embeddings for the vocoder.

        Args:
            codes: [T, 16] int64 - T frames, 16 codebooks

        Returns: [512, T] float32 - vocoder input format

        Vectorized: batch lookup + single matmul per codebook instead of per-frame loop.
        """
        T = codes.shape[0]
        embeddings = np.zeros((T, 512), dtype=np.float32)

        for cb_idx in range(NUM_CODE_GROUPS):
            cb_codes = codes[:, cb_idx].astype(np.intp)  # [T]
            cb_embeds = self._codebook_embeds[cb_idx][cb_codes]  # [T, 256]
            proj = self._output_proj_first if cb_idx == 0 else self._output_proj_rest
            embeddings += cb_embeds @ proj.T  # [T, 256] @ [256, 512] -> [T, 512]

        return embeddings.T  # [512, T]

    def _decode_audio(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode codec codes to audio using vocoder.

        Args:
            codes_array: [T, 16] int64
        """
        total_frames = codes_array.shape[0]
        if total_frames == 0:
            return np.zeros(0, dtype=np.float32)

        if self._vocoder_type == "stream":
            return self._decode_audio_stream(codes_array)
        else:
            return self._decode_audio_noembed(codes_array)

    def _decode_audio_stream(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode using tokenizer12hz_decode_stream.rknn (takes [1, 75, 16] int64 codes).

        Decodes in chunks of 25 frames with 50-frame context.
        """
        total_frames = codes_array.shape[0]
        chunk_size = 25
        ctx_size = 50
        total_T = ctx_size + chunk_size  # 75 (fixed input size)
        audio_chunks = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            actual_chunk = end - start
            ctx_start = max(0, start - ctx_size)

            # Get context + chunk codes
            chunk_codes = codes_array[ctx_start:end]  # [ctx+chunk, 16]

            # Pad to fixed size [75, 16]
            padded = np.zeros((total_T, 16), dtype=np.int64)
            padded[:chunk_codes.shape[0]] = chunk_codes

            # Reload RKNN context (workaround for RKNPU driver bug: Conv layer
            # in decoder.4/block.2/conv2 hangs after first inference per context)
            self._reload_vocoder()

            # Run vocoder
            vocoder_input = padded[np.newaxis, :, :]  # [1, 75, 16]
            outputs = self._vocoder.inference(inputs=[vocoder_input])
            audio_raw = np.array(outputs[0]).flatten()

            # Extract only the chunk audio (skip context audio)
            ctx_frames_used = start - ctx_start
            ctx_samples = ctx_frames_used * SAMPLES_PER_FRAME
            chunk_samples = actual_chunk * SAMPLES_PER_FRAME

            audio_chunk = audio_raw[ctx_samples:ctx_samples + chunk_samples]
            audio_chunks.append(audio_chunk)

        return np.concatenate(audio_chunks).astype(np.float32)

    def _decode_audio_noembed(self, codes_array: np.ndarray) -> np.ndarray:
        """Decode using decoder_ctx25_int8.rknn (takes [1, 512, 50] float32 embeddings).

        Converts codes to embeddings first, then decodes in chunks of 25 frames
        with 25-frame context.
        """
        total_frames = codes_array.shape[0]

        # Convert codes to continuous embeddings
        all_embeddings = self._codes_to_embeddings(codes_array)  # [512, T]

        chunk_size = 25
        ctx_size = VOCODER_CTX_FRAMES  # 25
        total_T = ctx_size + chunk_size  # 50 (vocoder fixed input size)
        n_chunks = (total_frames + chunk_size - 1) // chunk_size
        logger.info("Vocoder: %d frames -> %d chunks", total_frames, n_chunks)
        audio_chunks = []

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            actual_chunk = end - start
            ctx_start = max(0, start - ctx_size)

            # Get context + chunk embeddings
            chunk_emb = all_embeddings[:, ctx_start:end]  # [512, ctx+chunk]

            # Pad to fixed size [512, 50]
            padded = np.zeros((512, total_T), dtype=np.float32)
            padded[:, :chunk_emb.shape[1]] = chunk_emb

            # Reload RKNN context (workaround for RKNPU driver bug: Conv layer
            # in decoder.4/block.2/conv2 hangs after first inference per context)
            self._reload_vocoder()

            # Run vocoder
            vocoder_input = padded[np.newaxis, :, :]  # [1, 512, 50]
            chunk_idx = start // chunk_size
            logger.info("Vocoder chunk %d/%d (frames %d-%d)", chunk_idx + 1, n_chunks, start, end)
            outputs = self._vocoder.inference(inputs=[vocoder_input])
            logger.info("Vocoder chunk %d/%d done", chunk_idx + 1, n_chunks)
            audio_raw = np.array(outputs[0]).flatten()

            # Extract only the chunk audio (skip context audio)
            ctx_frames_used = start - ctx_start
            ctx_samples = ctx_frames_used * SAMPLES_PER_FRAME
            chunk_samples = actual_chunk * SAMPLES_PER_FRAME

            audio_chunk = audio_raw[ctx_samples:ctx_samples + chunk_samples]
            audio_chunks.append(audio_chunk)

        return np.concatenate(audio_chunks).astype(np.float32)

    # ── Cleanup ──────────────────────────────────────────────────

    def cleanup(self):
        """Release all NPU/RKLLM resources to prevent zombie threads."""
        self._ready = False

        if self._talker is not None:
            try:
                self._talker.destroy()
                logger.info("RKLLM talker destroyed")
            except Exception as e:
                logger.warning("Failed to destroy RKLLM talker: %s", e)
            self._talker = None

        for attr, name in [
            ("_text_project", "text_project"),
            ("_codec_embed", "codec_embed"),
            ("_code_predictor", "code_predictor"),
            ("_code_predictor_embed", "code_predictor_embed"),
            ("_vocoder", "vocoder"),
        ]:
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.release()
                    logger.info("RKNN %s released", name)
                except Exception as e:
                    logger.warning("Failed to release RKNN %s: %s", name, e)
                setattr(self, attr, None)

        if self._cp_engine is not None:
            try:
                self._cp_engine.destroy()
                logger.info("C engine destroyed")
            except Exception as e:
                logger.warning("Failed to destroy C engine: %s", e)
            self._cp_engine = None

        logger.info("TTSService cleanup complete")

    # ── Streaming ────────────────────────────────────────────────

    def _vocode_chunk(
        self,
        all_codes: list[list[int]],
        start_frame: int,
        num_frames: int,
        all_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Vocode frames [start_frame:start_frame+num_frames] with context.

        Args:
            all_codes: full list of frame codes accumulated so far (unused when
                       all_embeddings is provided; kept for API clarity)
            start_frame: index of the first NEW frame in all_embeddings
            num_frames: number of new frames to vocode
            all_embeddings: [512, T] embeddings for ALL frames generated so far

        Returns: float32 audio for the num_frames new frames only
        """
        ctx_start = max(0, start_frame - VOCODER_CTX_FRAMES)
        end = start_frame + num_frames

        chunk_emb = all_embeddings[:, ctx_start:end]  # [512, ctx+chunk]

        # Pad to fixed size [512, 50]
        total_T = VOCODER_CTX_FRAMES + 25  # 50
        padded = np.zeros((512, total_T), dtype=np.float32)
        padded[:, :chunk_emb.shape[1]] = chunk_emb

        # Reload RKNN context (workaround for RKNPU driver bug: Conv layer
        # in decoder.4/block.2/conv2 hangs after first inference per context)
        self._reload_vocoder()

        vocoder_input = padded[np.newaxis, :, :]  # [1, 512, 50]
        outputs = self._vocoder.inference(inputs=[vocoder_input])
        audio_raw = np.array(outputs[0]).flatten()

        # Extract only the new-frame audio (skip context)
        ctx_frames_used = start_frame - ctx_start
        ctx_samples = ctx_frames_used * SAMPLES_PER_FRAME
        chunk_samples = num_frames * SAMPLES_PER_FRAME
        return audio_raw[ctx_samples:ctx_samples + chunk_samples].astype(np.float32)

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        temperature: float = 0.8,
        top_k: int = 5,
        max_new_tokens: int = 300,
        first_chunk_frames: int = 10,
        chunk_frames: int = 25,
        language: str = "chinese",
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """Stream TTS: yields (audio_chunk_float32, metadata) tuples.

        First yield comes after first_chunk_frames AR frames (low TTFT).
        Subsequent yields come every chunk_frames AR frames.
        Only supported when vocoder_type == 'noembed' (decoder_ctx25_int8).
        """
        if self._vocoder_type != "noembed":
            # Fallback: synthesize everything and yield as single chunk
            wav_bytes, meta = self.synthesize(
                text, speaker_id=speaker_id, speed=speed,
                temperature=temperature, top_k=top_k,
                max_new_tokens=max_new_tokens, language=language,
            )
            buf = io.BytesIO(wav_bytes)
            audio, _ = sf.read(buf, dtype="float32")
            yield audio, meta
            return

        t_start = time.perf_counter()

        # Step 0: Reload talker if destroyed by previous non-streaming call
        if self._talker is None:
            self._load_rkllm_talker()

        # Step 1: Tokenize
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._tokenizer.encode(formatted_text)

        # Step 2: Build prefill embeddings (official format)
        prefill_embeds, tts_pad_vec = self._build_prefill(input_ids, language=language)

        # Step 3: Talker prefill (keep_history=1 so KV survives for decode steps)
        self._talker.clear_kv_cache()
        result = self._talker.run_embed(prefill_embeds, mode=1, keep_history=1)
        hidden = result["hidden"]
        last_hidden = hidden[-1:]  # [1, 1024]

        # Step 4: Initial logits
        logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]

        t_prefill_done = time.perf_counter()

        # Step 5: AR generate loop — accumulate codes and vocode progressively
        all_codes: list[list[int]] = []
        all_embeddings: Optional[np.ndarray] = None  # [512, T] grown incrementally
        min_new_tokens = 2
        last_vocoded_frame = 0   # how many frames have been vocoded so far
        is_first_chunk = True
        target_frames = first_chunk_frames  # current chunk target

        # EOS bias for quantized model
        text_chars = len(text.strip())
        est_frames = max(text_chars * FRAMES_PER_CHAR, 10)
        max_new_tokens = min(max_new_tokens, max(text_chars * FRAMES_PER_CHAR * 3, 30))

        chunk_index = 0

        def _flush_chunk(up_to_frame: int) -> None:
            """Vocode frames [last_vocoded_frame:up_to_frame] and yield."""
            nonlocal last_vocoded_frame, all_embeddings, chunk_index
            num_new = up_to_frame - last_vocoded_frame
            if num_new <= 0:
                return

            # Compute embeddings for newly generated frames only, then extend
            new_codes_array = np.array(
                all_codes[last_vocoded_frame:up_to_frame], dtype=np.int64
            )  # [num_new, 16]
            new_emb = self._codes_to_embeddings(new_codes_array)  # [512, num_new]

            if all_embeddings is None:
                all_embeddings = new_emb
            else:
                all_embeddings = np.concatenate([all_embeddings, new_emb], axis=1)

            audio_chunk = self._vocode_chunk(
                all_codes, last_vocoded_frame, num_new, all_embeddings
            )

            elapsed = time.perf_counter() - t_start
            meta = {
                "chunk_index": chunk_index,
                "frames": num_new,
                "start_frame": last_vocoded_frame,
                "elapsed": round(elapsed, 3),
            }
            chunk_index += 1
            last_vocoded_frame = up_to_frame
            return audio_chunk, meta

        # Suppress control tokens [2048, 3072) except EOS
        suppress_mask = np.ones(TALKER_VOCAB_SIZE, dtype=bool)
        suppress_mask[:CODE_PREDICTOR_VOCAB_SIZE] = False
        suppress_mask[CODEC_EOS_TOKEN_ID] = False
        suppress_indices = np.where(suppress_mask)[0]

        for step in range(max_new_tokens):
            # Suppress control tokens (always)
            logits[suppress_indices] = -float("inf")

            # Mask EOS for first two frames
            if step < min_new_tokens:
                logits[CODEC_EOS_TOKEN_ID] = -float("inf")

            # Apply progressive EOS bias after estimated duration
            if step >= est_frames:
                ramp = min((step - est_frames) / EOS_BIAS_RAMP_FRAMES, 1.0)
                eos_bias = ramp * EOS_BIAS_MAX
                logits[CODEC_EOS_TOKEN_ID] += eos_bias

            # Greedy EOS check
            if step >= min_new_tokens and np.argmax(logits) == CODEC_EOS_TOKEN_ID:
                logger.info("Stream EOS at step %d (greedy)", step)
                break

            primary_code = self._sample_top_k(
                logits, top_k=top_k, temperature=temperature,
                eos_id=CODEC_EOS_TOKEN_ID if step >= min_new_tokens else None,
            )

            if primary_code == CODEC_EOS_TOKEN_ID:
                logger.info("Stream EOS at step %d", step)
                break

            # Primary embedding
            primary_embed = self._run_codec_embed([primary_code])  # [1, 1024]

            # Residual codes
            frame_codes = [primary_code]
            if self._cp_engine is not None:
                n_residual = NUM_CODE_GROUPS - 1
                codes, codec_sum = self._cp_engine.run(last_hidden[0], primary_embed[0], num_steps=n_residual)
                frame_codes.extend(int(c) for c in codes)
            else:
                codec_sum = primary_embed[0].copy()
                latest_embed = primary_embed[0]
                for j in range(NUM_CODE_GROUPS - 1):
                    cp_input = np.stack([last_hidden[0], latest_embed])[np.newaxis, :, :]
                    cp_logits = self._run_code_predictor(cp_input)
                    cp_logits_last = cp_logits[0, 0]
                    res_code = self._sample_top_k(
                        cp_logits_last[:CODE_PREDICTOR_VOCAB_SIZE],
                        top_k=1, temperature=0.0,
                    )
                    frame_codes.append(res_code)
                    res_embed = self._run_code_predictor_embed(res_code, j)
                    latest_embed = res_embed
                    codec_sum += res_embed

            all_codes.append(frame_codes)
            current_frames = len(all_codes)

            # Build next talker input (official format: tts_pad + codec_sum)
            next_embed = (tts_pad_vec + codec_sum)[np.newaxis, :]  # [1, 1024]

            # Talker decode step (keep_history=1 = append to KV cache from prefill)
            result = self._talker.run_embed(next_embed, mode=1, keep_history=1)
            last_hidden = result["hidden"][-1:]  # [1, 1024]
            logits = (last_hidden @ self._codec_head_weight.T)[0]  # [3072]

            # Check if we should emit a chunk
            if current_frames >= last_vocoded_frame + target_frames:
                chunk_result = _flush_chunk(current_frames)
                if chunk_result is not None:
                    audio_chunk, meta = chunk_result
                    if speed != 1.0 and speed > 0:
                        n_orig = len(audio_chunk)
                        n_new = int(n_orig / speed)
                        indices = np.linspace(0, n_orig - 1, n_new)
                        audio_chunk = np.interp(
                            indices, np.arange(n_orig), audio_chunk
                        ).astype(np.float32)
                    yield audio_chunk, meta
                # Switch to regular chunk size after first emission
                if is_first_chunk:
                    is_first_chunk = False
                    target_frames = chunk_frames

        # Flush any remaining frames
        remaining = len(all_codes) - last_vocoded_frame
        if remaining > 0:
            chunk_result = _flush_chunk(len(all_codes))
            if chunk_result is not None:
                audio_chunk, meta = chunk_result
                if speed != 1.0 and speed > 0:
                    n_orig = len(audio_chunk)
                    n_new = int(n_orig / speed)
                    indices = np.linspace(0, n_orig - 1, n_new)
                    audio_chunk = np.interp(
                        indices, np.arange(n_orig), audio_chunk
                    ).astype(np.float32)
                yield audio_chunk, meta

        total_time = time.perf_counter() - t_start
        logger.info(
            "Stream TTS done: %d frames in %.1fs (%.1f fps)",
            len(all_codes), total_time,
            len(all_codes) / total_time if total_time > 0 else 0,
        )

    # ── WAV ──────────────────────────────────────────────────────

    @staticmethod
    def _make_wav(audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ── Module-level singleton ───────────────────────────────────────

_service: Optional[TTSService] = None


def preload():
    global _service
    model_dir = os.environ.get("MODEL_DIR", "/opt/tts/models")
    _service = TTSService(model_dir)
    _service.load()


def is_ready() -> bool:
    return _service is not None and _service.is_ready()


def synthesize(
    text: str,
    speaker_id: int = 0,
    speed: float = 1.0,
    pitch_shift: float = None,
) -> tuple[bytes, dict]:
    if _service is None:
        raise RuntimeError("TTS service not loaded")
    return _service.synthesize(text, speaker_id=speaker_id, speed=speed or 1.0)


def get_sample_rate() -> int:
    if _service is None:
        return SAMPLE_RATE
    return _service.get_sample_rate()


def synthesize_stream(
    text: str,
    speaker_id: int = 0,
    speed: float = 1.0,
    pitch_shift: float = None,
) -> Iterator[tuple[np.ndarray, dict]]:
    if _service is None:
        raise RuntimeError("TTS service not loaded")
    yield from _service.synthesize_stream(text, speaker_id=speaker_id, speed=speed or 1.0)
