"""
RKNN Encoder wrapper for Qwen3-ASR.

Supports two modes (auto-detected at init):
  - Merged encoder: Single RKNN model per size (faster, recommended)
  - Split encoder:  Frontend + Backend pair per size (legacy)

Auto-selects the best model size for each audio chunk to minimize NPU compute.

Merged model naming:
  qwen3_asr_encoder_merged.fp16.{Ns}.rk3576.rknn
  Input:  input_features(1, 128, T_mel) + attention_mask(1, 1, T_down, T_down)
  Output: audio_embeds(1, T_down, 1024)

Split model naming:
  qwen3_asr_encoder_frontend.fp16.{Ns}.rk3576.rknn
  qwen3_asr_encoder_backend.fp32.{Ns}.rk3576.rknn
Models without size suffix are treated as the default (30s).
"""

import os
import re
import time
import numpy as np
from .mel import MelExtractor
from .config import SAMPLE_RATE


class RknnEncoder:
    """
    RKNN audio encoder with multi-size model support.

    Scans for available models and auto-detects merged vs split mode.
    Merged mode is preferred when available (single inference call, faster).
    """

    def __init__(self, encoder_dir: str, mel_filter_path: str,
                 npu_core_mask: str = "NPU_CORE_0_1",
                 sizes: list = None):
        """
        Args:
            encoder_dir:     Directory containing .rknn encoder files
            mel_filter_path: Path to mel_filters.npy
            npu_core_mask:   NPU core allocation ("NPU_CORE_0", "NPU_CORE_0_1")
            sizes:           List of sizes (seconds) to load. None = load all.
                             Example: [2, 4] loads only 2s and 4s models.
        """
        from rknnlite.api import RKNNLite

        self.mel = MelExtractor(mel_filter_path)
        self._core_mask_str = npu_core_mask
        core = getattr(RKNNLite, npu_core_mask, RKNNLite.NPU_CORE_0_1)

        # Scan for model files in encoder_dir and parent directory
        merged_files = {}    # {sec: path}
        fe_files = {}        # {sec: path}
        be_files = {}        # {sec: path}

        search_dirs = [encoder_dir]
        parent = os.path.dirname(encoder_dir)
        if parent and parent != encoder_dir:
            search_dirs.append(parent)
        # Also check grandparent (model_dir itself when encoder_dir = models/encoder/rk3576)
        grandparent = os.path.dirname(parent) if parent else None
        if grandparent and grandparent != parent:
            search_dirs.append(grandparent)

        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".rknn"):
                    continue
                fp = os.path.join(d, f)
                m = re.search(r'\.(\d+)s\.', f)
                sec = int(m.group(1)) if m else 30
                if "merged" in f:
                    merged_files.setdefault(sec, fp)
                elif "frontend" in f:
                    fe_files.setdefault(sec, fp)
                elif "backend" in f:
                    be_files.setdefault(sec, fp)

        # Filter by requested sizes
        if sizes is not None:
            wanted = set(sizes)
            merged_files = {k: v for k, v in merged_files.items() if k in wanted}
            fe_files = {k: v for k, v in fe_files.items() if k in wanted}
            be_files = {k: v for k, v in be_files.items() if k in wanted}

        # Prefer merged mode
        if merged_files:
            self._mode = "merged"
            self._models = {}
            for sec in sorted(merged_files.keys()):
                model = RKNNLite(verbose=False)
                ret = model.load_rknn(merged_files[sec])
                assert ret == 0, f"load_rknn failed: {merged_files[sec]}"
                assert model.init_runtime(core_mask=core) == 0
                mel_frames = sec * 100
                max_tokens = self._compute_token_len(mel_frames)
                self._models[sec] = (model, mel_frames, max_tokens)
            tag = "merged"
        else:
            # Fall back to split mode
            paired = sorted(set(fe_files.keys()) & set(be_files.keys()))
            if not paired:
                raise FileNotFoundError(
                    f"No encoder .rknn found. Searched: {search_dirs}")
            self._mode = "split"
            self._models = {}
            for sec in paired:
                fe = RKNNLite(verbose=False)
                assert fe.load_rknn(fe_files[sec]) == 0
                assert fe.init_runtime(core_mask=core) == 0
                be = RKNNLite(verbose=False)
                assert be.load_rknn(be_files[sec]) == 0
                assert be.init_runtime(core_mask=core) == 0
                mel_frames = sec * 100
                max_tokens = self._compute_token_len(mel_frames)
                self._models[sec] = (fe, be, mel_frames, max_tokens)
            tag = "split(FE+BE)"

        self._sizes = sorted(self._models.keys())
        self.embed_dim = 1024

        size_str = "/".join(f"{s}s" for s in self._sizes)
        print(f"[Encoder] {tag}, {len(self._sizes)} sizes: {size_str}")

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def _select_model(self, audio_seconds: float):
        """Select smallest model that can hold the audio."""
        for sec in self._sizes:
            if audio_seconds <= sec:
                return sec
        return self._sizes[-1]

    def encode(self, audio_chunk: np.ndarray) -> tuple:
        """
        Encode audio chunk to embedding tokens.

        Args:
            audio_chunk: float32 waveform at 16kHz

        Returns:
            tuple: (hidden, enc_ms, model_sec)
                hidden:    (N_tokens, 1024) float32 embedding
                enc_ms:    encoding time in milliseconds
                model_sec: which model size was used
        """
        actual_samples = len(audio_chunk)
        actual_seconds = actual_samples / SAMPLE_RATE
        model_sec = self._select_model(actual_seconds)

        if self._mode == "merged":
            return self._encode_merged(audio_chunk, model_sec, actual_seconds)
        else:
            return self._encode_split(audio_chunk, model_sec, actual_seconds)

    # ------------------------------------------------------------------ #
    # Merged encoder (single model)                                       #
    # ------------------------------------------------------------------ #

    def _encode_merged(self, audio_chunk, model_sec, actual_seconds):
        model, mel_frames, max_tokens = self._models[model_sec]
        max_samples = int(model_sec * SAMPLE_RATE)

        # Pad / clip audio
        if len(audio_chunk) < max_samples:
            audio_chunk = np.pad(audio_chunk, (0, max_samples - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:max_samples]

        # Mel: (128, T_mel)
        mel = self.mel(audio_chunk)
        if mel.shape[1] < mel_frames:
            mel = np.pad(mel, [(0, 0), (0, mel_frames - mel.shape[1])])
        else:
            mel = mel[:, :mel_frames]

        # Merged input: (1, 128, mel_frames), channels-first
        mel_input = mel[np.newaxis, ...].astype(np.float32)

        # Causal attention mask: all zeros = no masking
        n_tok = max_tokens
        mask = np.zeros((1, 1, n_tok, n_tok), dtype=np.float32)

        t0 = time.monotonic()
        hidden = model.inference(inputs=[mel_input, mask])[0]
        enc_ms = (time.monotonic() - t0) * 1000

        if hidden.ndim == 3:
            hidden = hidden[0]   # (T_down, 1024)

        # Trim to actual token count
        actual_frames = min(int(actual_seconds * 100), mel_frames)
        actual_tokens = min(self._compute_token_len(actual_frames),
                            hidden.shape[0])
        if actual_tokens < hidden.shape[0]:
            hidden = hidden[:actual_tokens]

        return hidden.astype(np.float32), enc_ms, model_sec

    # ------------------------------------------------------------------ #
    # Split encoder (frontend + backend)                                  #
    # ------------------------------------------------------------------ #

    def _encode_split(self, audio_chunk, model_sec, actual_seconds):
        fe, be, mel_frames, max_tokens = self._models[model_sec]
        max_samples = int(model_sec * SAMPLE_RATE)

        if len(audio_chunk) < max_samples:
            audio_chunk = np.pad(audio_chunk, (0, max_samples - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:max_samples]

        # Mel: (128, T_mel)  →  transpose to (T_mel, 128) for split FE
        mel = self.mel(audio_chunk)
        mel_input = mel.T
        if mel_input.shape[0] < mel_frames:
            pad = np.zeros((mel_frames - mel_input.shape[0], 128),
                           dtype=mel_input.dtype)
            mel_input = np.concatenate([mel_input, pad], axis=0)
        else:
            mel_input = mel_input[:mel_frames]
        mel_input = mel_input[np.newaxis, ...]   # (1, mel_frames, 128)

        t0 = time.monotonic()
        feat_out = fe.inference(inputs=[mel_input])[0]
        hidden = be.inference(inputs=[feat_out])[0]
        enc_ms = (time.monotonic() - t0) * 1000

        if hidden.ndim == 3:
            hidden = hidden[0]

        actual_frames = min(int(actual_seconds * 100), mel_frames)
        actual_tokens = min(self._compute_token_len(actual_frames),
                            hidden.shape[0])
        if actual_tokens < hidden.shape[0]:
            hidden = hidden[:actual_tokens]

        return hidden.astype(np.float32), enc_ms, model_sec

    # ------------------------------------------------------------------ #
    # Properties & helpers                                                #
    # ------------------------------------------------------------------ #

    @property
    def mode(self):
        """Current encoder mode: 'merged' or 'split'."""
        return self._mode

    @property
    def available_sizes(self):
        """Available model sizes in seconds."""
        return list(self._sizes)

    @property
    def max_seconds(self):
        """Max audio length supported (by largest model)."""
        return self._sizes[-1]

    @staticmethod
    def _compute_token_len(n_frames: int, chunk_size: int = 100,
                           tokens_per_chunk: int = 13) -> int:
        """Compute output token count from input mel frames."""
        full = n_frames // chunk_size
        rem = n_frames % chunk_size
        if rem == 0:
            return full * tokens_per_chunk
        f = (rem - 1) // 2 + 1
        f = (f - 1) // 2 + 1
        f = (f - 1) // 2 + 1
        return full * tokens_per_chunk + f

    def release(self):
        """Release all RKNN resources."""
        for sec in list(self._models.keys()):
            tup = self._models[sec]
            if self._mode == "merged":
                model, _, _ = tup
                try:
                    model.release()
                except Exception:
                    pass
            else:
                fe, be, _, _ = tup
                try:
                    fe.release()
                except Exception:
                    pass
                try:
                    be.release()
                except Exception:
                    pass
        self._models.clear()
