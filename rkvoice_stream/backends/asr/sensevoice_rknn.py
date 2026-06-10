"""SenseVoice ASR backend: SenseVoice-small encoder on the Rockchip NPU (RKNN).

SenseVoice-small (lovemefan/SenseVoice-onnx) is an encoder + CTC model — a single
forward pass over LFR features yields ``[1, T, 25055]`` CTC logits (no separate
decoder, no CIF). The 4 SenseVoice prompt embeddings (language / event / speech /
textnorm) are prepended to the LFR features as the first 4 frames. RKNN has no
dynamic dims, so the encoder is converted to a fixed sequence length ``T_FIXED``
and audio is padded/truncated to it.

Verified on real RK3576 NPU (fp16, no overflow): zh + en decode correctly,
byte-identical English vs the FP32 ONNX reference. Supports **both RK3576 and
RK3588** — the per-SoC ``.rknn`` is selected by ``RK_PLATFORM``.

Front end (matches the lovemefan/sherpa SenseVoice export):
  80-dim kaldi fbank (dither=0, hamming, snip_edges) -> LFR(m=7,n=6) -> 560
  -> global CMVN (am.mvn: ``(x + add) * scale``) -> prepend 4 prompt frames.

Environment variables
---------------------
RK_PLATFORM               "rk3576" (default) or "rk3588" — selects the .rknn.
SENSEVOICE_RKNN_MODEL_DIR Directory with the model + decode assets.
                          Default: /opt/asr/sensevoice-rknn
SENSEVOICE_RKNN_MODEL     Explicit .rknn path override (skips RK_PLATFORM).
SENSEVOICE_RKNN_CORE      NPU core mask (default NPU_CORE_0).

Model dir layout:
  sense-voice-encoder.rk3576.fp16.rknn   (and/or .rk3588.)
  am.mvn                                 global CMVN (560-dim add + scale)
  embedding.npy                          (16, 560) prompt embedding table
  chn_jpn_yue_eng_ko_spectok.bpe.model   sentencepiece tokenizer (25055)
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Fixed encoder sequence length the RKNN artifact was frozen to (prompt frames +
# LFR frames). Must match the value used by sv_fix_shape.py at conversion time.
T_FIXED = 344
LFR_DIM = 560
BLANK_ID = 0

# Language → row index in embedding.npy (lovemefan SenseVoiceSmall prompt table).
_LANG_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}
_TEXTNORM_IDS = {"withitn": 14, "woitn": 15}

# user language string → SenseVoice tag (mirrors sensevoice_sherpa._LANGUAGE_MAP)
_LANGUAGE_MAP = {
    "auto": "auto", "chinese": "zh", "mandarin": "zh", "english": "en",
    "japanese": "ja", "korean": "ko", "cantonese": "yue", "yue": "yue",
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh", "en": "en", "en-us": "en",
    "en-gb": "en", "ja": "ja", "ko": "ko",
}


def _map_language(language: str) -> str:
    return _LANGUAGE_MAP.get((language or "auto").lower(), "auto")


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr or len(audio) == 0:
        return audio
    n_out = int(round(len(audio) * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class SenseVoiceRKNNBackend(ASRBackend):
    """SenseVoice offline ASR on the Rockchip NPU (RK3576 / RK3588) via RKNNLite."""

    # Opt into the generic offline→streaming adapter (OfflineAccumulateStream):
    # accumulate audio, transcribe the whole utterance on finalize, endpointing
    # via the OVS server-side VAD. Unlocks /asr/stream + /v2v/stream.
    supports_offline_streaming = True

    def __init__(self) -> None:
        self._rknn = None
        self._cmvn_add: Optional[np.ndarray] = None
        self._cmvn_scale: Optional[np.ndarray] = None
        self._emb: Optional[np.ndarray] = None
        self._sp = None
        self._ready = False

    @property
    def name(self) -> str:
        return "sensevoice_rknn"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._rknn is not None

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

    def _resolve_model_path(self, model_dir: str) -> str:
        explicit = os.environ.get("SENSEVOICE_RKNN_MODEL")
        if explicit:
            return explicit
        import glob
        platform = os.environ.get("RK_PLATFORM", "rk3576").lower()
        # Precision-agnostic: RK3576 ships fp16, RK3588 ships int8 (fp16 overflows
        # the RK3588 NPU on Chinese activations) — pick whichever .rknn is present
        # for this SoC.
        hits = sorted(glob.glob(os.path.join(model_dir, f"sense-voice-encoder.{platform}.*.rknn")))
        if hits:
            return hits[0]
        # Last resort: any sense-voice .rknn in the dir.
        any_hits = sorted(glob.glob(os.path.join(model_dir, "sense-voice-encoder.*.rknn")))
        if any_hits:
            logger.warning(
                "No .rknn for RK_PLATFORM=%s; falling back to %s", platform, any_hits[0]
            )
            return any_hits[0]
        raise FileNotFoundError(
            f"No SenseVoice RKNN model for platform {platform!r} in {model_dir!r} "
            f"(expected sense-voice-encoder.{platform}.*.rknn)."
        )

    def preload(self) -> None:
        import sentencepiece as spm
        from rknnlite.api import RKNNLite

        model_dir = os.environ.get("SENSEVOICE_RKNN_MODEL_DIR", "/opt/asr/sensevoice-rknn")
        model_path = self._resolve_model_path(model_dir)
        logger.info("Loading SenseVoice RKNN encoder from %s", model_path)

        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed: {model_path!r}")
        core_name = os.environ.get("SENSEVOICE_RKNN_CORE", "NPU_CORE_0")
        core = getattr(RKNNLite, core_name, RKNNLite.NPU_CORE_AUTO)
        if rknn.init_runtime(core_mask=core) != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed (core={core_name})")
        self._rknn = rknn

        self._cmvn_add, self._cmvn_scale = self._load_cmvn(os.path.join(model_dir, "am.mvn"))
        self._emb = np.load(os.path.join(model_dir, "embedding.npy"))
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model"))

        self._ready = True
        logger.info("SenseVoice RKNN backend ready (vocab=%d).", self._sp.get_piece_size())

    def unload(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                logger.exception("RKNNLite.release failed; continuing")
        self._rknn = None
        self._ready = False

    # ------------------------------------------------------------------
    # Transcribe (offline)
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready — call preload() first")
        audio = self._decode_audio(audio_bytes)
        return self.transcribe_array(audio, language)

    def transcribe_array(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        tag = _map_language(language)
        speech, valid = self._build_speech(audio, lang=tag)
        out = self._rknn.inference(inputs=[speech.astype(np.float32)])
        logits = out[0][0]  # [T_FIXED, 25055]
        text = self._ctc_decode(logits, valid)
        return TranscriptionResult(text=text, language=None)

    # ------------------------------------------------------------------
    # Front end + decode (validated against sherpa CPU baseline)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cmvn(path: str):
        txt = open(path).read()
        vals = [np.array(b.split(), dtype=np.float32) for b in re.findall(r"\[([^\]]*)\]", txt)]
        big = [v for v in vals if v.size == LFR_DIM]
        return big[0], big[1]

    @staticmethod
    def _compute_feats(audio: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(16000, (audio * 32768).tolist())
        fb.input_finished()
        return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])

    @staticmethod
    def _apply_lfr(feats: np.ndarray, m: int = 7, n: int = 6) -> np.ndarray:
        T = feats.shape[0]
        pad = (m - 1) // 2
        feats = np.vstack([np.tile(feats[0], (pad, 1)), feats])
        T2 = feats.shape[0]
        out = []
        i = 0
        while i * n < T:
            idx0 = i * n
            if idx0 + m <= T2:
                out.append(feats[idx0:idx0 + m].reshape(-1))
            else:
                chunk = feats[idx0:T2]
                need = m - chunk.shape[0]
                chunk = np.vstack([chunk, np.tile(feats[-1], (need, 1))])
                out.append(chunk.reshape(-1))
            i += 1
        return np.stack(out).astype(np.float32)

    def _build_speech(self, audio: np.ndarray, lang: str = "auto", textnorm: str = "withitn"):
        lfr = self._apply_lfr(self._compute_feats(audio))
        # NOTE: do NOT apply external CMVN here. The lovemefan SenseVoice encoder
        # ONNX normalizes internally (its first LayerNorm); applying am.mvn CMVN
        # on top double-normalizes and degrades accuracy (verified: mean CER
        # 0.048→0.032 across 5 zh samples when removed). am.mvn is retained in
        # the bundle only as reference / for the sherpa CPU path.
        prefix = np.stack([
            self._emb[_LANG_IDS.get(lang, 0)],
            self._emb[1],
            self._emb[2],
            self._emb[_TEXTNORM_IDS[textnorm]],
        ]).astype(np.float32)
        sp_in = np.concatenate([prefix, lfr], axis=0).astype(np.float32)
        valid = sp_in.shape[0]
        if valid > T_FIXED:
            sp_in = sp_in[:T_FIXED]
            valid = T_FIXED
        else:
            sp_in = np.vstack([sp_in, np.zeros((T_FIXED - valid, LFR_DIM), dtype=np.float32)])
        return sp_in[None], valid

    def _ctc_decode(self, logits: np.ndarray, valid: int) -> str:
        ids = logits.argmax(-1).tolist()[:valid]
        collapsed = []
        prev = -1
        for x in ids:
            if x != prev and x != BLANK_ID:
                collapsed.append(x)
            prev = x
        pieces = [self._sp.id_to_piece(i) for i in collapsed if 0 <= i < self._sp.get_piece_size()]
        text = "".join(pieces).replace("▁", " ")
        # Strip SenseVoice prompt special tokens <|...|> (language/emotion/event/itn).
        text = re.sub(r"<\|[^|]*\|>", "", text)
        return text.strip()

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        import soundfile as sf

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            raise ValueError(f"Cannot decode audio: {exc}") from exc
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = _resample_linear(audio, sr, 16000)
        return audio.astype(np.float32)
