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
SENSEVOICE_RKNN_CORE      NPU core mask for the single-worker case
                          (default NPU_CORE_0).
SENSEVOICE_RKNN_WORKERS   Number of NPU worker contexts, one per NPU core
                          (default: 3 on rk3588, 2 on rk3576, 1 elsewhere).
                          Set to 1 to restore the previous single-context
                          behaviour.
SENSEVOICE_RKNN_T_FIXED   Encoder sequence length the .rknn was frozen to.
                          Normally left unset: it is read off the artifact
                          filename (``...t172.rknn`` -> 172), falling back to
                          344 for files that carry no ``tN`` token.

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
import queue
import re
import threading
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Fixed encoder sequence length the RKNN artifact was frozen to (4 prompt frames
# + LFR frames). Must match the value used by sv_fix_shape.py at conversion
# time, so it is read off the artifact rather than assumed: a filename carrying
# a ``tN`` token (``sense-voice-encoder.rk3588.fp16-scaled.t172.rknn``) pins N,
# and anything else keeps the original 344.
T_FIXED_DEFAULT = 344
_T_IN_NAME = re.compile(r"\.t(\d+)\.")
LFR_DIM = 560
BLANK_ID = 0

# NPU core each worker binds to, in build order. ``core_mask`` accepts only the
# driver's enum values ({0 AUTO, 1 core0, 2 core1, 4 core2, 3 core0+1,
# 7 core0+1+2}); a bitwise combination such as 5 or 6 is rejected outright
# ("The core mode 6 is not supported currently"), and NPU_CORE_AUTO binds a
# single core rather than spreading work. So each worker takes exactly one
# single-core enum and the pool provides the parallelism.
_WORKER_CORE_NAMES = ("NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2")

# Physical NPU cores per SoC — the ceiling on useful workers.
_PLATFORM_WORKERS = {"rk3588": 3, "rk3576": 2}

# Tombstone left in a retired pool so a caller already blocked in ``get()``
# wakes up and fails loudly instead of waiting on a queue nobody will refill.
# It is put back by whoever draws it, so one is enough for any number of them.
_POOL_CLOSED = object()


def _resolve_worker_count(platform: str) -> int:
    """Workers to attempt, from ``SENSEVOICE_RKNN_WORKERS`` else the SoC."""
    default = _PLATFORM_WORKERS.get(platform, 1)
    raw = os.environ.get("SENSEVOICE_RKNN_WORKERS", "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "SENSEVOICE_RKNN_WORKERS=%r is not an integer; using %d", raw, default
        )
        return default
    if n < 1:
        logger.warning("SENSEVOICE_RKNN_WORKERS=%d must be >= 1; using 1", n)
        return 1
    if n > len(_WORKER_CORE_NAMES):
        logger.warning(
            "SENSEVOICE_RKNN_WORKERS=%d exceeds the %d single-core masks the "
            "driver exposes; using %d",
            n, len(_WORKER_CORE_NAMES), len(_WORKER_CORE_NAMES),
        )
        return len(_WORKER_CORE_NAMES)
    return n

def _join_windows(parts: list) -> str:
    """Concatenate per-window decodes, spacing only where scripts need it."""
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if text and (text[-1].isascii() and text[-1].isalnum()) and (
            part[0].isascii() and part[0].isalnum()
        ):
            text += " "
        text += part
    return text


def _resolve_t_fixed(model_path: str) -> int:
    """Encoder length for this artifact: env override, else the filename."""
    raw = os.environ.get("SENSEVOICE_RKNN_T_FIXED", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            logger.warning("SENSEVOICE_RKNN_T_FIXED=%r is not an integer; ignoring", raw)
        else:
            if n > 4:
                return n
            logger.warning("SENSEVOICE_RKNN_T_FIXED=%d must exceed the 4 prompt frames; ignoring", n)
    m = _T_IN_NAME.search(os.path.basename(model_path))
    if m:
        return int(m.group(1))
    return T_FIXED_DEFAULT


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
        # One RKNNLite context per NPU core, plus a queue that hands exactly
        # one context to one caller at a time. ``queue.Queue`` is the pool:
        # borrowing blocks until a context is free, so more callers than
        # workers is correct (they wait), not a race.
        self._contexts: list = []
        self._worker_cores: list[str] = []
        self._pool: "Optional[queue.Queue]" = None
        # Guards every transition of ``_pool``/``_contexts`` and the
        # borrowed-context bookkeeping. Held only for pointer swaps and
        # counter updates — never across an inference or a release.
        self._lifecycle_lock = threading.Lock()
        self._borrowed = 0
        # Encoder length of the loaded artifact; the default stands until
        # preload() reads the real one off the file it opened.
        self._t_fixed = T_FIXED_DEFAULT
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
    def npu_worker_count(self) -> int:
        """Contexts actually built. 0 before ``preload()``."""
        return len(self._contexts)

    @property
    def npu_worker_cores(self) -> list[str]:
        """Core mask name of each built context, in build order."""
        return list(self._worker_cores)

    @property
    def supports_parallel(self) -> bool:
        return len(self._contexts) > 1

    @property
    def max_concurrent(self) -> int:
        return max(1, len(self._contexts))

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
        # Precision-agnostic: pick whichever .rknn is present for this SoC.
        # Both RK3576 and RK3588 ship `fp16-scaled` — plain fp16 overflows the
        # last encoder block's FFN on either NPU, and int8 collapses the
        # 25055-way CTC projection. sorted()[0] is what makes the transition
        # safe: `fp16-scaled.rknn` sorts before `fp16.rknn` ('-' < '.'), so a
        # directory still holding a legacy unscaled file picks the scaled one.
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
        t_fixed = _resolve_t_fixed(model_path)
        logger.info(
            "Loading SenseVoice RKNN encoder from %s (T_FIXED=%d)", model_path, t_fixed
        )

        with self._lifecycle_lock:
            if self._pool is not None:
                # Idempotent. Building a second pool over a live one would
                # publish contexts that a caller may already be holding, and
                # leave that caller returning to a queue nobody drains.
                logger.info("SenseVoice RKNN backend already loaded; skipping preload")
                return

        platform = os.environ.get("RK_PLATFORM", "rk3576").lower()
        want = _resolve_worker_count(platform)
        if want == 1:
            # Single worker keeps the historical env: an operator who pinned
            # the backend to one specific core still gets that core.
            core_names = [os.environ.get("SENSEVOICE_RKNN_CORE", "NPU_CORE_0")]
        else:
            core_names = list(_WORKER_CORE_NAMES[:want])

        # Built into locals and published in one step at the end, so a failure
        # anywhere below leaves the backend unloaded rather than half-loaded
        # with live contexts nobody will release.
        contexts: list = []
        cores: list[str] = []
        try:
            for core_name in core_names:
                core = getattr(RKNNLite, core_name, RKNNLite.NPU_CORE_AUTO)
                worker = RKNNLite(verbose=False)
                try:
                    rc_load = worker.load_rknn(model_path)
                except Exception:
                    self._discard(worker)
                    raise
                if rc_load != 0:
                    self._discard(worker)
                    if not contexts:
                        raise RuntimeError(f"RKNNLite.load_rknn failed: {model_path!r}")
                    logger.warning(
                        "SenseVoice RKNN worker on %s: load_rknn failed; "
                        "continuing with %d worker(s)", core_name, len(contexts),
                    )
                    break
                try:
                    rc_init = worker.init_runtime(core_mask=core)
                except Exception:
                    self._discard(worker)
                    raise
                if rc_init != 0:
                    self._discard(worker)
                    if not contexts:
                        raise RuntimeError(
                            f"RKNNLite.init_runtime failed (core={core_name})"
                        )
                    # A later core failing is a degraded start, not a dead one:
                    # serve at the width that did come up and say so.
                    logger.warning(
                        "SenseVoice RKNN worker on %s: init_runtime failed; "
                        "continuing with %d worker(s)", core_name, len(contexts),
                    )
                    break
                contexts.append(worker)
                cores.append(core_name)

            cmvn_add, cmvn_scale = self._load_cmvn(os.path.join(model_dir, "am.mvn"))
            emb = np.load(os.path.join(model_dir, "embedding.npy"))
            sp = spm.SentencePieceProcessor()
            sp.load(os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
        except BaseException:
            for worker in contexts:
                self._discard(worker)
            raise

        pool: queue.Queue = queue.Queue()
        for worker in contexts:
            pool.put(worker)
        with self._lifecycle_lock:
            if self._pool is not None:
                # Another thread won the race and published first; drop ours.
                logger.warning("SenseVoice RKNN pool already published; discarding this one")
                stale = contexts
            else:
                stale = []
                self._contexts = contexts
                self._worker_cores = cores
                self._pool = pool
                self._borrowed = 0
                # Kept for callers (and tests) that reach for a single context.
                self._rknn = contexts[0]
                self._cmvn_add, self._cmvn_scale = cmvn_add, cmvn_scale
                self._emb = emb
                self._sp = sp
                self._t_fixed = t_fixed
                self._ready = True
        for worker in stale:
            self._discard(worker)
        if stale:
            return

        logger.info(
            "SenseVoice RKNN worker pool: %d context(s) on %s (platform=%s, "
            "T_FIXED=%d, max %.1fs audio per encoder pass)",
            len(contexts), ", ".join(cores), platform, t_fixed,
            (t_fixed - 4) * 0.06,
        )
        logger.info("SenseVoice RKNN backend ready (vocab=%d).", sp.get_piece_size())

    @staticmethod
    def _discard(worker) -> None:
        """Release a context that failed to come up. Never raises."""
        try:
            worker.release()
        except Exception:
            logger.exception("RKNNLite.release failed on a half-built worker")

    def unload(self) -> None:
        """Retire the pool and release the contexts nobody is holding.

        Retiring is a single pointer swap under the lock, so a second
        ``unload()`` finds nothing and cannot release a handle twice — a double
        release inside the RKNN runtime takes the process down rather than
        raising. After the swap no borrower can put anything back, so what sits
        in the retired queue is exactly the set that is free: it is drained
        without blocking and released here.

        A context inside ``inference()`` right now is NOT released here —
        releasing under a running inference faults the runtime. Its borrower
        sees the retired pool when it finishes and releases it there, so every
        context is still released exactly once, just not all on this thread.
        """
        with self._lifecycle_lock:
            contexts, self._contexts = self._contexts, []
            pool, self._pool = self._pool, None
            in_flight = self._borrowed
            self._worker_cores = []
            self._rknn = None
            self._ready = False
        if pool is None:
            return
        free = []
        while True:
            try:
                item = pool.get_nowait()
            except queue.Empty:
                break
            if item is not _POOL_CLOSED:
                free.append(item)
        # Leave a tombstone so a caller already blocked in get(), or one that
        # arrives later holding a stale reference, wakes and fails loudly.
        pool.put(_POOL_CLOSED)
        if in_flight:
            logger.info(
                "SenseVoice RKNN unload: %d of %d context(s) still in flight;"
                " each is released by the caller holding it",
                in_flight, len(contexts),
            )
        for worker in free:
            try:
                worker.release()
            except Exception:
                logger.exception("RKNNLite.release failed; continuing")

    # ------------------------------------------------------------------
    # Transcribe (offline)
    # ------------------------------------------------------------------

    def _borrow(self):
        """Take one context out of the pool, blocking until one is free.

        Returns the pool it came from alongside it: the caller gives it back to
        *that* queue, not to whatever ``self._pool`` points at by then.
        """
        with self._lifecycle_lock:
            pool = self._pool
            if pool is None:
                raise RuntimeError("ASR backend not ready — call preload() first")
        worker = pool.get()
        if worker is _POOL_CLOSED:
            pool.put(_POOL_CLOSED)  # keep the tombstone for the next waiter
            raise RuntimeError("ASR backend was unloaded")
        with self._lifecycle_lock:
            if self._pool is not pool:
                # Unloaded between the get() and here: we are this context's
                # last owner, so retire it rather than leak it.
                self._discard(worker)
                raise RuntimeError("ASR backend was unloaded")
            self._borrowed += 1
        return pool, worker

    def _give_back(self, pool, worker) -> None:
        """Return a borrowed context, or release it if the pool was retired."""
        with self._lifecycle_lock:
            retired = self._pool is not pool
            self._borrowed = max(0, self._borrowed - 1)
            if not retired:
                pool.put(worker)
        if retired:
            try:
                worker.release()
            except Exception:
                logger.exception("RKNNLite.release failed; continuing")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready — call preload() first")
        audio = self._decode_audio(audio_bytes)
        return self.transcribe_array(audio, language)

    def transcribe_array(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        if self._pool is None:
            raise RuntimeError("ASR backend not ready — call preload() first")
        tag = _map_language(language)
        # Feature extraction is pure numpy and needs no context; only the NPU
        # call borrows one, so the pool is held for the shortest span.
        windows = self._build_speech(audio, lang=tag)
        # One borrow per window rather than one for the whole utterance: a long
        # utterance must not hold a context across several NPU passes while
        # other sessions queue behind it.
        parts = []
        for speech, valid in windows:
            pool, worker = self._borrow()
            try:
                out = worker.inference(inputs=[speech.astype(np.float32)])
            finally:
                self._give_back(pool, worker)
            logits = out[0][0]  # [T, 25055]
            parts.append(self._ctc_decode(logits, valid))
        return TranscriptionResult(text=_join_windows(parts), language=None)

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
        # External am.mvn CMVN. NOTE: on FP32 onnxruntime removing this *looks*
        # better (mean CER 0.048→0.032, lovemefan ONNX self-normalizes), BUT on
        # the real RK3576 fp16 NPU it is NOT a clean win — noCMVN/CMVN each
        # collapse on a different subset of short zh clips (per-sample fp16
        # instability). Kept = the device-validated default (passed offline +
        # streaming e2e). Revisit only with a proper on-device corpus CER.
        lfr = (lfr + self._cmvn_add) * self._cmvn_scale
        prefix = np.stack([
            self._emb[_LANG_IDS.get(lang, 0)],
            self._emb[1],
            self._emb[2],
            self._emb[_TEXTNORM_IDS[textnorm]],
        ]).astype(np.float32)
        return self._windows(lfr, prefix)

    def _windows(self, lfr: np.ndarray, prefix: np.ndarray) -> list:
        """Split LFR features into ``[1, T, 560]`` encoder inputs.

        The encoder is frozen to T frames, of which the 4 prompt frames are
        fixed overhead, so at most ``T - 4`` LFR frames fit in one pass. Audio
        longer than that used to be truncated here and the tail silently lost;
        it is now cut into consecutive windows that each carry their own prompt
        prefix and are decoded in order. Speech shorter than the window — the
        normal case for a VAD-delimited utterance — still yields exactly one
        window, unchanged.
        """
        t_fixed = self._t_fixed
        span = t_fixed - prefix.shape[0]
        out = []
        for start in range(0, max(lfr.shape[0], 1), span):
            chunk = lfr[start:start + span]
            sp_in = np.concatenate([prefix, chunk], axis=0).astype(np.float32)
            valid = sp_in.shape[0]
            if valid < t_fixed:
                sp_in = np.vstack(
                    [sp_in, np.zeros((t_fixed - valid, LFR_DIM), dtype=np.float32)]
                )
            out.append((sp_in[None], valid))
        return out

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
