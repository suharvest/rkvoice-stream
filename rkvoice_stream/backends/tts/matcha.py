"""
RKNN TTS Backend for RK3576

完整流程：
文本 → 文本前端 (sherpa-onnx, CPU) → tokens → Matcha RKNN (NPU) → mel → Vocos RKNN (NPU) → ISTFT (CPU) → 音频

Matcha acoustic model is compiled with fixed output shape (probe-first surgery).
Available bucket models:
  - matcha-s64.rknn:  seq_len=80,  x_len=64,  ~599 mel frames, ~9.6s, 53MB, ~430ms
  - matcha-s140.rknn: seq_len=160, x_len=140, ~1278 mel frames, ~20s, 60MB, ~900ms

Vocos vocoder compiled with fixed TIME_FRAMES:
  - vocos-16khz-600.rknn: 600 frames input, ~26.9MB, ~80ms

性能 (s64 + vocos-600, typical 50-token sentence):
- 文本前端: <10ms (CPU 查表)
- Matcha RKNN: ~430ms (NPU, s64 bucket)
- Vocos RKNN: ~80ms (NPU, 600 frames)
- ISTFT: ~50ms (CPU)
- 总计: ~570ms for 7.6s audio, RTF ~0.07
"""

from __future__ import annotations

import os
import re
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional

# 音频参数
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
MAX_SEQ_LEN = int(os.environ.get('MATCHA_MAX_PHONEMES', '64'))
# RKNN model compiled input sequence length (must match the shape used during
# rknn.build()).  All tensor inputs to the Matcha RKNN model are padded to
# this length so that byte sizes match the static-shape expectations.
# Default 80 matches the s64 bucket model (seq_len=80, x_len=64, ~599 mel frames).
MATCHA_MODEL_SEQ_LEN = int(os.environ.get('MATCHA_MODEL_SEQ_LEN', '80'))
# Vocos model compiled time frame dimension (must match vocos RKNN build).
VOCOS_FRAMES = int(os.environ.get('VOCOS_FRAMES', '600'))

# Split model constants (encoder + estimator with CPU FP32 ODE loop)
MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978
# ODE constants for split RKNN mode (not used in ORT mode).
# The default runtime step count is 1 (env MATCHA_ODE_STEPS=1) for best
# FP16 precision; N_ODE_STEPS=3 is only used for loading time_emb files.
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3  # number of pre-computed time_emb files (always 3)
MAX_FRAMES = 600
TIME_EMB_DIM = 256
N_TIME_BLOCKS = 6


# ---------------------------------------------------------------------------
# English text frontend
#
# Reverse-engineered from sherpa-onnx's native frontend for this exact model
# (matcha-icefall-zh-en) by dumping the int64 tensor it feeds the acoustic
# model (OfflineTtsModelConfig(debug=True) -> offline-tts-matcha-impl.h:467).
# The rules below reproduce 83/83 probe texts token-for-token.
#
# The previous implementation emitted raw IPA characters, which drove the
# model with phones outside its training distribution: `aɪ` went in as `a`+`ɪ`
# rather than the single token `I` that tokens.txt reserves for it.  Measured
# effect on English word error rate: 38.3% -> 0.0%.
# ---------------------------------------------------------------------------

# Greedy longest-first.  Note `ɚ` is an *expansion*: one IPA char, two tokens.
_IPA_MAP: dict[str, tuple[str, ...]] = {
    'aɪ': ('I',),
    'eɪ': ('A',),
    'oʊ': ('O',),
    'aʊ': ('W',),
    'ɔɪ': ('Y',),
    'tʃ': ('ʧ',),
    'dʒ': ('ʤ',),
    'ɚ': ('ə', 'ɹ'),
}
_IPA_MAX_KEY = max(len(k) for k in _IPA_MAP)

# Length marks are always dropped; stress marks are kept in place.
_IPA_STRIP = frozenset('ː')

# Punctuation is normalized before lookup.  `:` folds onto `,` -- token 3
# exists in tokens.txt but sherpa never emits it.
_PUNCT_NORMALIZE = {
    '：': ',', ':': ',', '，': ',', '、': ',',
    '。': '.', '！': '!', '？': '?', '；': ';',
}
# Dropped outright (they also swallow the word boundary around them).
_PUNCT_DROP = frozenset("-'（）《》")
# Cutting a chunk here means a separate acoustic-model call, matching
# sherpa's per-sentence chunking.  `;` `—` `…` `(` `)` do NOT cut.
_PUNCT_CHUNK_END = frozenset(',.!?"“”')

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*")
_ITEM_RE = re.compile(
    r"[一-鿿]+"                              # CJK run
    r"|[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]*"  # latin word
    r"|\s+"
    r"|."                                            # punct, digit or symbol
)


class _EspeakPhonemizer:
    """Per-word IPA via the espeak-ng C API.

    The CLI is deliberately not used: `espeak-ng --ipa -q -- <word>` promotes
    stress on a lone function word, so `the` comes back as `ðˈə` where sherpa
    (and the C API) produce `ðə`, and `but`/`our` get `ˈ` instead of `ˌ`.
    Measured on 83 words: the CLI disagrees with sherpa on 12 of them purely
    from this effect.  espeak is not thread-safe and initializes globally, so
    this is a process-wide singleton behind a lock.
    """

    _ESPEAK_CHARS_UTF8 = 1
    _ESPEAK_PHONEMES_IPA = 2
    _AUDIO_OUTPUT_RETRIEVAL = 1

    _instance: '_EspeakPhonemizer | None' = None
    _instance_lock = threading.Lock()

    def __init__(self, data_dir: Optional[str] = None):
        import ctypes

        self._lock = threading.Lock()
        self._lib = ctypes.CDLL('libespeak-ng.so.1')
        self._lib.espeak_Initialize.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.espeak_Initialize.restype = ctypes.c_int
        self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self._lib.espeak_SetVoiceByName.restype = ctypes.c_int
        self._lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
        ]
        self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p

        path = data_dir.encode('utf-8') if data_dir and os.path.isdir(data_dir) else None
        rate = self._lib.espeak_Initialize(self._AUDIO_OUTPUT_RETRIEVAL, 0, path, 0)
        if rate < 0:
            raise RuntimeError(f"espeak_Initialize failed (rc={rate})")
        if self._lib.espeak_SetVoiceByName(b'en-us') != 0:
            raise RuntimeError("espeak_SetVoiceByName('en-us') failed")

    @classmethod
    def get(cls, data_dir: Optional[str] = None) -> '_EspeakPhonemizer | None':
        with cls._instance_lock:
            if cls._instance is None:
                try:
                    cls._instance = cls(data_dir)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "espeak-ng C API unavailable (%s) — falling back to the "
                        "CLI; function-word stress will differ slightly from "
                        "the reference frontend", e,
                    )
                    cls._instance = False  # sentinel: tried and failed
            return cls._instance or None

    def phonemize(self, word: str) -> str:
        import ctypes

        with self._lock:
            buf = ctypes.c_char_p(word.encode('utf-8'))
            ptr = ctypes.cast(ctypes.byref(buf), ctypes.POINTER(ctypes.c_void_p))
            out = []
            while True:
                res = self._lib.espeak_TextToPhonemes(
                    ptr, self._ESPEAK_CHARS_UTF8, self._ESPEAK_PHONEMES_IPA,
                )
                if res:
                    out.append(res.decode('utf-8'))
                if not buf.value:
                    break
            return ''.join(out)


def ipa_to_token_strings(ipa: str) -> list[str]:
    """Map an espeak IPA string onto this model's token alphabet.

    Greedy longest match over _IPA_MAP, length marks dropped, everything else
    (stress marks included) passed through verbatim.  Spaces inside a single
    word's espeak output become word-boundary tokens -- espeak expands e.g. an
    emoji into several words.
    """
    out: list[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        for width in range(min(_IPA_MAX_KEY, n - i), 0, -1):
            key = ipa[i:i + width]
            if key in _IPA_MAP:
                out.extend(_IPA_MAP[key])
                i += width
                break
        else:
            ch = ipa[i]
            i += 1
            if ch in _IPA_STRIP:
                continue
            if ch == ' ':
                out.append(' ')
            elif ch.isspace():
                continue
            else:
                out.append(ch)
    return out


# Mel frames each class of token is worth, fitted against the ORT path's true
# frame count over 74 measurements on RK3576 (37 texts x both tokenizations).
#
# The formula this replaces was `11.9 * num_tokens + 51`, which treated every
# token alike.  11.9 is the *Chinese pinyin* rate: it was calibrated on Chinese
# only, and the docstring's own calibration points are reproduced by the
# pinyin-only law to within a couple of frames.  Applied to English -- where a
# phoneme costs 4.7 frames, not 11.9 -- it overshot by 2.5x, and the estimate
# then clamped at the model's output width, so every long English segment
# rendered the full padded window (measured: 19.3s of speech stretched to
# 34.6s).  MAE against ORT truth: 240 frames before, 11.7 after.
#
# Punctuation is the most expensive class, not a free one -- those tokens buy
# real pauses (A/B: seven commas were worth 139 frames).  Word boundaries are
# free to within measurement noise.
_MEL_FRAMES_PER_TOKEN = {
    'phoneme': 4.7,    # English/IPA
    'pinyin': 11.6,    # Chinese syllable
    'punct': 16.4,     # a rendered pause
    'boundary': 0.0,   # fitted at 0.022
}
_MEL_FRAMES_CONST = 40.0

_PINYIN_RE = re.compile(r'^[a-z]+[1-5]$')


def classify_token(token: str) -> str:
    """Which cost class a token string belongs to (see _MEL_FRAMES_PER_TOKEN)."""
    if token == ' ':
        return 'boundary'
    if len(token) == 1 and not token.isalnum():
        return 'punct'
    if _PINYIN_RE.match(token):
        return 'pinyin'
    return 'phoneme'


def utterance_gain(audio: np.ndarray) -> tuple[float, bool]:
    """Playback gain for one utterance, returned as (gain, needs_clip).

    Plain peak normalization lets a single transient decide the level for the
    whole utterance: a 3-sample ISTFT spike once ducked a 4 s clip by 23 dB.
    When the peak is a clear outlier (p99.9 far below it), normalize against
    p99.9 and clip the outlier instead of ducking everything.

    Returned as a factor rather than applied in place so the streaming path
    can compute it once for an utterance and reuse it across every segment.
    Normalizing each segment independently would make the level pump between
    sentences of the same reply.
    """
    if len(audio) == 0:
        return 1.0, False
    peak = float(np.abs(audio).max())
    if peak <= 0:
        return 1.0, False
    p999 = float(np.percentile(np.abs(audio), 99.9))
    if p999 > 0 and p999 / peak < 0.25:
        return 0.95 / p999, True
    return 0.95 / peak, False


def parse_tokens_file(path: str) -> dict[str, int]:
    """Parse an icefall ``tokens.txt`` into {token: id}.

    Each line is ``<token> <id>`` -- and the token may itself be a space.
    Line 1 of matcha-icefall-zh-en/tokens.txt is literally ``"  1"``: the
    word-boundary token (id 1) that sherpa-onnx requires via
    ``token2id_.at(" ")`` (matcha-tts-lexicon.cc:265).

    The original parse was ``line.strip().split()`` with the id taken as
    ``i + 1``.  That swallowed the leading space, so the boundary token became
    unreachable and a phantom token ``"1"`` was registered in its place --
    English then had no way to mark word boundaries at all.  Split the id off
    from the right, keep the token verbatim, and trust the file's own id
    column instead of assuming line order.
    """
    token_to_id: dict[str, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            token, sep, id_str = line.rpartition(' ')
            if not sep:
                continue
            try:
                token_to_id[token] = int(id_str)
            except ValueError:
                continue
    return token_to_id


def _npu_lock():
    """Shared host-NPU lock, or None when the ASR backend is unavailable.

    Same accessor the qwen3_rknn TTS backend uses (qwen3_rknn.py:53-57): the
    RKLLM ASR decoder holds all three NPU cores while decoding, so any RKNN
    context here has to serialize against it regardless of core pinning.
    """
    try:
        from rkvoice_stream.backends.asr.qwen3_rk import get_npu_lock
    except ImportError:
        return None
    return get_npu_lock()


class RKNNMatchaVocoder:
    """RKNN 加速的 Matcha TTS 引擎"""

    def __init__(
        self,
        matcha_rknn_path: str,
        vocos_rknn_path: str,
        lexicon_path: str,
        tokens_path: str,
        data_dir: str,
    ):
        self.matcha_rknn_path = matcha_rknn_path
        self.vocos_rknn_path = vocos_rknn_path
        self.lexicon_path = lexicon_path
        self.tokens_path = tokens_path
        self.data_dir = data_dir

        # 加载后的模型
        self._matcha = None
        self._matcha_backend = None  # 'rknn', 'rknn_split', or 'ort'
        self._matcha_encoder = None   # split mode: encoder RKNN
        self._matcha_estimator = None  # split mode: estimator RKNN
        self._cstsin_refs = None      # ctypes refs for CstSin custom op (prevent GC)
        self._time_emb_steps = None   # split mode: [3, 6, 256] time embeddings
        # Probed from single RKNN model at load time: (x_seq_len, noise_shape, has_x_length)
        # noise_shape = None means the model takes x_length instead of a noise tensor
        self._matcha_rknn_input_meta = None
        self._vocos = None
        self._lexicon = None
        self._token_to_id = None
        self._id_to_class = None   # token id -> mel-cost class, built lazily

    def load(self):
        """加载所有模型和资源"""
        import logging
        log = logging.getLogger(__name__)

        from rknnlite.api import RKNNLite

        # 加载 lexicon
        self._lexicon = {}
        with open(self.lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self._lexicon[parts[0]] = parts[1:]

        # 加载 tokens
        self._token_to_id = parse_tokens_file(self.tokens_path)
        self._id_to_class = None

        # 加载 Matcha 声学模型
        # Priority: 1) split RKNN (best FP16 precision), 2) single RKNN, 3) ORT fallback
        matcha_dir = os.path.dirname(self.matcha_rknn_path)
        split_dir = os.environ.get('MATCHA_SPLIT_DIR',
                                   os.path.join(matcha_dir, 'matcha-split'))
        enc_path = os.path.join(split_dir, 'matcha-encoder-fp16.rknn')
        est_path = os.path.join(split_dir, 'matcha-estimator-fp16.rknn')
        te_path = os.path.join(split_dir, 'time_emb_step0.npy')

        use_ort = os.environ.get('MATCHA_USE_ORT', '').lower() in ('1', 'true', 'yes')

        # Try split RKNN mode first (best precision: ODE loop on CPU FP32)
        if not use_ort and os.path.exists(enc_path) and os.path.exists(est_path) and os.path.exists(te_path):
            try:
                self._matcha_encoder = RKNNLite(verbose=False)
                ret = self._matcha_encoder.load_rknn(enc_path)
                if ret != 0:
                    raise RuntimeError(f"load encoder ret={ret}")
                ret = self._matcha_encoder.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init encoder runtime ret={ret}")

                self._matcha_estimator = RKNNLite(verbose=False)
                ret = self._matcha_estimator.load_rknn(est_path)
                if ret != 0:
                    raise RuntimeError(f"load estimator ret={ret}")
                ret = self._matcha_estimator.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init estimator runtime ret={ret}")

                # Register custom CPU ops if the .so exists
                cstops_so = os.environ.get(
                    'CSTOPS_LIB', '/opt/tts/lib/libcstops.so')
                if os.path.exists(cstops_so):
                    from rkvoice_stream.backends.custom_ops.rknn_custom_ops import register_custom_ops
                    self._cstsin_refs = register_custom_ops(
                        self._matcha_estimator,
                        lib_path=cstops_so,
                    )
                    if self._cstsin_refs is None:
                        log.warning("Custom op registration failed")
                else:
                    self._cstsin_refs = None

                self._time_emb_steps = [
                    np.load(os.path.join(split_dir, f'time_emb_step{i}.npy'))
                    for i in range(N_ODE_STEPS)
                ]
                self._matcha_backend = 'rknn_split'
                log.info("Matcha loaded via split RKNN (encoder+estimator, CPU FP32 ODE)")
            except Exception as e:
                log.warning("Matcha split RKNN load failed (%s), trying single RKNN", e)
                for m in (self._matcha_encoder, self._matcha_estimator):
                    if m is not None:
                        try:
                            m.release()
                        except Exception:
                            pass
                self._matcha_encoder = self._matcha_estimator = self._time_emb_steps = None

        # Try single RKNN
        if self._matcha_backend is None and not use_ort and os.path.exists(self.matcha_rknn_path):
            try:
                self._matcha = RKNNLite(verbose=False)
                ret = self._matcha.load_rknn(self.matcha_rknn_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn ret={ret}")
                ret = self._matcha.init_runtime(core_mask=1)
                if ret != 0:
                    raise RuntimeError(f"init_runtime ret={ret}")
                self._matcha_backend = 'rknn'
                # Probe input shapes so run_matcha can build correctly-shaped arrays.
                # Input layout may be:
                #   v1 (old): [x, x_length, noise_scale, length_scale]
                #   v2 (new): [x, noise_scale, length_scale, noise]
                try:
                    rt = self._matcha.rknn_runtime
                    attrs = [rt.get_tensor_attr(i, False) for i in range(4)]
                    names = [a.name.decode() if isinstance(a.name, bytes) else a.name
                             for a in attrs]
                    x_seq_len = int(list(attrs[0].dims[:attrs[0].n_dims])[-1])
                    if 'noise' in names:
                        noise_idx = names.index('noise')
                        noise_dims = list(attrs[noise_idx].dims[:attrs[noise_idx].n_dims])
                        self._matcha_rknn_input_meta = {
                            'x_seq_len': x_seq_len,
                            'noise_shape': tuple(noise_dims),  # e.g. (1, 80, 256)
                            'layout': 'v2',  # [x, noise_scale, length_scale, noise]
                        }
                    else:
                        self._matcha_rknn_input_meta = {
                            'x_seq_len': x_seq_len,
                            'noise_shape': None,
                            'layout': 'v1',  # [x, x_length, noise_scale, length_scale]
                        }
                    log.info("Matcha RKNN input meta: %s", self._matcha_rknn_input_meta)
                except Exception as probe_err:
                    log.warning("Matcha RKNN input probe failed (%s), using env defaults", probe_err)
                    self._matcha_rknn_input_meta = None
                log.info("Matcha acoustic model loaded via RKNN (single model)")
            except Exception as e:
                log.warning("Matcha RKNN load failed (%s), trying ORT fallback", e)
                if self._matcha is not None:
                    try:
                        self._matcha.release()
                    except Exception:
                        pass
                self._matcha = None

        # ORT fallback
        if self._matcha_backend is None:
            matcha_onnx_path = self.matcha_rknn_path.replace('.rknn', '.onnx')
            if not os.path.exists(matcha_onnx_path):
                alt = os.path.join(matcha_dir, 'model-steps-3.onnx')
                if os.path.exists(alt):
                    matcha_onnx_path = alt
                else:
                    matcha_onnx_path = os.environ.get('MATCHA_ONNX_PATH', matcha_onnx_path)
            if os.path.exists(matcha_onnx_path):
                import onnxruntime as ort
                self._matcha = ort.InferenceSession(
                    matcha_onnx_path,
                    providers=['CPUExecutionProvider'],
                )
                self._matcha_backend = 'ort'
                log.info("Matcha acoustic model loaded via ORT (CPU): %s", matcha_onnx_path)

        if self._matcha_backend is None:
            raise RuntimeError(
                f"无法加载 Matcha 声学模型: split={split_dir}, "
                f"RKNN={self.matcha_rknn_path}"
            )

        # 加载 Vocos RKNN
        self._vocos = RKNNLite(verbose=False)
        ret = self._vocos.load_rknn(self.vocos_rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载 Vocos RKNN 失败: ret={ret}")
        ret = self._vocos.init_runtime(core_mask=1)  # NPU_CORE_0 only; CORE_1 reserved for ASR encoder
        if ret != 0:
            raise RuntimeError(f"初始化 Vocos RKNN 运行时失败: ret={ret}")

        # Probe the vocoder's real time capacity.
        #
        # Neither the filename nor VOCOS_FRAMES can be trusted: the shipped
        # "vocos-16khz-600.rknn" is in fact a 256-frame build, and rknn-lite
        # does NOT error on an oversized input — it silently reinterprets the
        # buffer and emits garbage (measured: -22 dB vs a correctly sized
        # call).  Ask the model what it actually is.
        self._vocos_frames = VOCOS_FRAMES
        for probe_frames in (VOCOS_FRAMES, MAX_FRAMES):
            try:
                probe = np.zeros((1, 80, probe_frames), dtype=np.float32)
                lock = _npu_lock()
                if lock is not None:
                    with lock:
                        out = self._vocos.inference(inputs=[probe])
                else:
                    out = self._vocos.inference(inputs=[probe])
                actual = int(np.asarray(out[0]).shape[-1])
            except Exception as e:  # undersized buffer can legitimately fail
                log.debug("Vocos capacity probe at %d frames failed: %s", probe_frames, e)
                continue
            if actual > 0:
                self._vocos_frames = actual
                if actual != VOCOS_FRAMES:
                    log.warning(
                        "Vocos model %s emits %d frames, but VOCOS_FRAMES=%d — "
                        "using the model's %d (%.2fs per chunk). Oversized "
                        "inputs are silently misread by rknn-lite, so the env "
                        "value is ignored.",
                        os.path.basename(self.vocos_rknn_path), actual,
                        VOCOS_FRAMES, actual, actual * HOP_LENGTH / SAMPLE_RATE,
                    )
                else:
                    log.info(
                        "Vocos capacity: %d frames (%.2fs per chunk)",
                        actual, actual * HOP_LENGTH / SAMPLE_RATE,
                    )
                break
        else:
            log.warning(
                "Vocos capacity probe failed — assuming VOCOS_FRAMES=%d",
                VOCOS_FRAMES,
            )

    def release(self):
        """释放资源"""
        for m in (self._matcha, self._matcha_encoder, self._matcha_estimator):
            if m is not None:
                try:
                    m.release()
                except Exception:
                    pass
        self._matcha = self._matcha_encoder = self._matcha_estimator = None
        self._cstsin_refs = None
        self._time_emb_steps = None
        self._matcha_backend = None
        if self._vocos:
            try:
                self._vocos.release()
            except Exception:
                pass
            self._vocos = None

    def _espeak_ipa(self, word: str) -> str:
        """Raw IPA for one word, preferring the C API over the CLI."""
        import logging

        engine = _EspeakPhonemizer.get(self.data_dir)
        if engine is not None:
            try:
                return engine.phonemize(word)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "espeak C API failed on %r (%s) — using the CLI", word, e,
                )

        import subprocess

        log = logging.getLogger(__name__)
        try:
            cmd = ["espeak-ng", "--ipa", "-v", "en-us", "-q", "--", word]
            env = os.environ.copy()
            if self.data_dir and os.path.isdir(self.data_dir):
                env["ESPEAK_DATA_PATH"] = self.data_dir
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, env=env,
            )
            if result.returncode != 0:
                log.warning(
                    "espeak-ng failed (rc=%d): %s",
                    result.returncode, result.stderr.strip(),
                )
                return ""
            return result.stdout.strip()
        except FileNotFoundError:
            log.warning("espeak-ng not found — English text will be skipped")
            return ""
        except subprocess.TimeoutExpired:
            log.warning("espeak-ng timed out on %r", word)
            return ""

    def _phonemize_english(self, word: str) -> list[str]:
        """One English word -> token strings, following the reference frontend.

        Resolution order matches sherpa's ConvertWordToIds: a word that is
        itself a token in the table bypasses espeak entirely (that is why "a"
        stays `a` rather than becoming `eɪ`/`A`).
        """
        if word in self._token_to_id:
            return [word]
        ipa = self._espeak_ipa(word)
        if not ipa:
            return []
        return ipa_to_token_strings(ipa)

    def text_to_tokens(self, text: str) -> list[int]:
        """
        将文本转换为 token IDs

        中文：lexicon 查表 → phonemes → token IDs
        英文：espeak-ng C API 逐词 → IPA → token 映射
        标点：归一化后作为 token 发出
        词边界：拉丁词之后插入空格 token(id 1)

        Mirrors sherpa-onnx's frontend for this model.  The old version emitted
        raw IPA characters and dropped both word boundaries and punctuation,
        leaving English as one unbroken phone run -- measured at 38.3% WER
        against 0.0% for the reference frontend.
        """
        # Build (is_latin_word, token_strings) items first, then assemble: the
        # word-boundary rule needs to know whether the *next* item emits
        # anything.
        items: list[tuple[bool, list[str]]] = []

        for raw in _ITEM_RE.findall(text):
            if not raw or raw.isspace():
                continue

            if _CJK_RE.match(raw):
                items.append((False, self._chinese_token_strings(raw)))
                continue

            if _WORD_RE.fullmatch(raw):
                items.append((True, self._phonemize_english(raw)))
                continue

            # Single non-word char: punctuation, digit or symbol.
            ch = _PUNCT_NORMALIZE.get(raw, raw)
            if ch in _PUNCT_DROP:
                continue
            if not ch.isalnum() and ch in self._token_to_id:
                items.append((False, [ch]))
                continue
            # Digits and stray symbols: let espeak say them out loud rather
            # than dropping them silently, as the old code did.
            items.append((False, self._phonemize_english(raw)))

        boundary = self._token_to_id.get(' ')
        tokens: list[int] = []
        for idx, (is_word, strings) in enumerate(items):
            for s in strings:
                tid = self._token_to_id.get(s)
                if tid is not None:
                    tokens.append(tid)
            if (
                is_word
                and strings
                and boundary is not None
                and idx + 1 < len(items)
                and items[idx + 1][1]
            ):
                tokens.append(boundary)

        return tokens

    def _chinese_token_strings(self, text: str) -> list[str]:
        """Chinese run -> token strings via longest-match lexicon lookup."""
        out: list[str] = []
        i = 0
        while i < len(text):
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i + length]
                if word in self._lexicon:
                    out.extend(self._lexicon[word])
                    i += length
                    break
            else:
                i += 1
        return out

    def _chinese_to_tokens(self, text: str) -> list[int]:
        """Convert Chinese text to token IDs via lexicon lookup."""
        tokens = []
        i = 0
        while i < len(text):
            # 尝试匹配最长词
            found = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self._lexicon:
                    phonemes = self._lexicon[word]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                    i += length
                    found = True
                    break

            if not found:
                # 单字处理
                char = text[i]
                if char in self._lexicon:
                    phonemes = self._lexicon[char]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                i += 1

        return tokens

    def run_matcha(
        self,
        tokens: list[int],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        运行 Matcha RKNN 声学模型

        Args:
            tokens: 音素 token IDs
            noise_scale: 噪声缩放因子
            length_scale: 时长缩放因子

        Returns:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数
        """
        num_tokens = len(tokens)
        x_length = np.array([num_tokens], dtype=np.int64)
        noise_scale_arr = np.array([noise_scale], dtype=np.float32)
        length_scale_arr = np.array([length_scale], dtype=np.float32)

        if self._matcha_backend == 'rknn_split':
            # Split mode: encoder (NPU) + estimator (NPU) + ODE loop (CPU FP32)
            tokens_padded = np.zeros((1, MATCHA_MODEL_SEQ_LEN), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            enc_out = self._matcha_encoder.inference(
                inputs=[tokens_padded, x_length, noise_scale_arr, length_scale_arr]
            )
            mu = enc_out[0]     # [1, 80, 600]
            mask = enc_out[1]   # [1, 1, 600]
            z = enc_out[2]      # [1, 80, 600] (z0 = noise * noise_scale)

            # ODE loop on CPU (FP32 precision).
            # 1-step Euler (dt=1.0) is preferred on RK3576: fewer FP16
            # accumulation errors and 2.6x faster than 3-step.
            n_steps = int(os.environ.get('MATCHA_ODE_STEPS', '1'))
            dt = np.float32(1.0 / n_steps)
            for step in range(n_steps):
                te = self._time_emb_steps[min(step, len(self._time_emb_steps) - 1)]
                feeds = [z, mu, mask]
                for i in range(N_TIME_BLOCKS):
                    feeds.append(te[i].reshape(1, TIME_EMB_DIM, 1).astype(np.float32))
                v = self._matcha_estimator.inference(inputs=feeds)[0]
                z = z + dt * v

            # Denormalize
            mel = z * np.float32(MEL_SIGMA) + np.float32(MEL_MEAN)

        elif self._matcha_backend == 'rknn':
            meta = self._matcha_rknn_input_meta
            if meta is not None and meta['layout'] == 'v2':
                # Newer exported model: [x, noise_scale, length_scale, noise]
                # x_length is not an explicit input; noise is pre-generated.
                seq_len = meta['x_seq_len']
                noise_shape = meta['noise_shape']
                tokens_padded = np.zeros((1, seq_len), dtype=np.int64)
                tokens_padded[0, :num_tokens] = tokens
                noise = np.random.randn(*noise_shape).astype(np.float32)
                mel = self._matcha.inference(
                    inputs=[tokens_padded, noise_scale_arr, length_scale_arr, noise]
                )[0]
            else:
                # Older exported model: [x, x_length, noise_scale, length_scale]
                tokens_padded = np.zeros((1, MATCHA_MODEL_SEQ_LEN), dtype=np.int64)
                tokens_padded[0, :num_tokens] = tokens
                mel = self._matcha.inference(
                    inputs=[tokens_padded, x_length, noise_scale_arr, length_scale_arr]
                )[0]
        else:
            # ORT fallback — dynamic shapes, no padding needed
            tokens_padded = np.zeros((1, max(num_tokens, 1)), dtype=np.int64)
            tokens_padded[0, :num_tokens] = tokens
            mel = self._matcha.run(
                None,
                {
                    'x': tokens_padded,
                    'x_length': x_length,
                    'noise_scale': noise_scale_arr,
                    'length_scale': length_scale_arr,
                },
            )[0]

        # Determine valid mel frames.
        T = mel.shape[2]
        if self._matcha_backend == 'ort':
            # ORT produces dynamic output — all frames are valid.
            mel_frames = T
        else:
            # RKNN produces fixed-size output (e.g., 600 frames for split, 599
            # for single), so the valid frame count has to be estimated.
            #
            # Margin is multiplicative plus a floor: 20% of a short utterance
            # is a thin cushion, and under-predicting truncates speech while
            # over-predicting only renders a little extra tail.  Over the 74
            # calibration measurements this leaves 16 frames of headroom in
            # the worst case (6 without the floor) and never under-predicts.
            est = int(
                self._estimate_mel_frames(tokens) * length_scale * 1.2 + 10 + 0.5
            )
            mel_frames = min(est, T)
            # Clamp to ORT-observed range as safety measure.
            mel = np.clip(mel, -25.0, 8.0)

        return mel, mel_frames

    def _estimate_mel_frames(self, tokens: list[int]) -> float:
        """Predict how many mel frames `tokens` will occupy.

        Only the RKNN acoustic paths need this -- ORT reports the true count.
        Costs are per token *class*, because a Chinese syllable, an English
        phoneme and a punctuation pause are worth wildly different durations.
        """
        if self._id_to_class is None:
            self._id_to_class = {
                tid: classify_token(tok)
                for tok, tid in (self._token_to_id or {}).items()
            }
        total = _MEL_FRAMES_CONST
        for tid in tokens:
            cls = self._id_to_class.get(tid, 'phoneme')
            total += _MEL_FRAMES_PER_TOKEN[cls]
        return total

    def run_vocos(self, mel: np.ndarray, mel_frames: int) -> np.ndarray:
        """
        运行 Vocos RKNN 声码器

        Args:
            mel: Mel 频谱图 [1, 80, T]
            mel_frames: 有效帧数

        Returns:
            audio: 音频样本
        """
        total_frames = min(mel_frames, mel.shape[2])
        if total_frames <= 0:
            return np.zeros(0, dtype=np.float32)

        # The vocoder has a fixed compiled window (self._vocos_frames).  Longer
        # utterances are rendered in overlapping chunks and stitched in the
        # *spectral* domain, so a single ISTFT still runs over the whole
        # utterance and there are no seams in the waveform.
        #
        # Previously anything past the window was simply dropped — a 447-frame
        # Chinese sentence rendered only 256 frames, losing 42% of its content.
        mag, x, y = self._run_vocos_frames(mel, total_frames)

        # ISTFT
        audio = self._istft(mag, x, y)

        # 裁剪到正确长度
        return audio[:total_frames * HOP_LENGTH]

    def _vocos_infer(self, mel_window: np.ndarray) -> list:
        """One vocoder call, serialized against the ASR backend's NPU use.

        The RKLLM decoder runs with npu_core_num=3 (all cores), so it overlaps
        this vocos context even though vocos is pinned to NPU_CORE_0.  Lock only
        the RKNN call — the ISTFT is pure numpy on the CPU and must stay
        outside.
        """
        lock = _npu_lock()
        if lock is not None:
            with lock:
                return self._vocos.inference(inputs=[mel_window])
        return self._vocos.inference(inputs=[mel_window])

    def _run_vocos_frames(
        self,
        mel: np.ndarray,
        total_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render `total_frames` mel frames through the fixed-window vocoder.

        Returns the concatenated (mag, cos, sin) spectra, exactly
        `total_frames` wide.  Chunks carry `ctx` frames of context on each side
        which are then discarded, so the convolutional receptive field is fed
        properly across chunk boundaries.
        """
        cap = int(getattr(self, '_vocos_frames', VOCOS_FRAMES) or VOCOS_FRAMES)

        def _emit(window: np.ndarray, lo: int, hi: int):
            buf = np.zeros((1, 80, cap), dtype=np.float32)
            n = window.shape[2]
            buf[:, :, :n] = window
            out = self._vocos_infer(buf)
            return (out[0][0][:, lo:hi], out[1][0][:, lo:hi], out[2][0][:, lo:hi])

        if total_frames <= cap:
            mag, x, y = _emit(mel[:, :, :total_frames], 0, total_frames)
            return mag, x, y

        ctx = min(32, cap // 8)
        stride = cap - 2 * ctx
        if stride <= 0:  # pathologically small window
            ctx, stride = 0, cap

        mags, xs, ys = [], [], []
        pos = 0
        while pos < total_frames:
            w_end = min(total_frames, max(pos - ctx, 0) + cap)
            w_start = max(0, w_end - cap)
            keep_end = min(total_frames, pos + stride)
            m, cx, sy = _emit(
                mel[:, :, w_start:w_end], pos - w_start, keep_end - w_start
            )
            mags.append(m)
            xs.append(cx)
            ys.append(sy)
            pos = keep_end

        return (
            np.concatenate(mags, axis=1),
            np.concatenate(xs, axis=1),
            np.concatenate(ys, axis=1),
        )

    def _istft(
        self,
        mag: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        逆短时傅里叶变换

        Args:
            mag: 幅度谱 [513, T]
            x: 余弦分量 (实部)
            y: 正弦分量 (虚部)

        Returns:
            audio: 重建的音频
        """
        # 重建复数频谱
        complex_spec = mag * (x + 1j * y)

        n_frames = complex_spec.shape[1]
        output_len = (n_frames - 1) * HOP_LENGTH + N_FFT

        audio = np.zeros(output_len, dtype=np.float32)
        window = np.hanning(N_FFT)

        # 重叠相加
        for i in range(n_frames):
            frame = np.fft.irfft(complex_spec[:, i], n=N_FFT) * window
            start = i * HOP_LENGTH
            audio[start:start + N_FFT] += frame

        # 归一化
        window_sum = np.zeros(output_len, dtype=np.float32)
        for i in range(n_frames):
            start = i * HOP_LENGTH
            window_sum[start:start + N_FFT] += window ** 2

        # Overlap-add normalization.  The outermost N_FFT - HOP_LENGTH samples
        # are covered by fewer than the full set of windows, so window_sum
        # tapers to ~0 at both edges.  Dividing there amplifies whatever the
        # vocoder emitted into a huge transient — the old
        # `np.maximum(window_sum, 1e-8)` turned the final samples into a spike
        # that then hijacked the peak normalization in synthesize() and ducked
        # the whole utterance by 20-30 dB.  Zero the degenerate edge instead of
        # dividing by a near-zero denominator.
        steady = float(np.median(window_sum[window_sum > 0])) if np.any(window_sum > 0) else 0.0
        floor = max(steady * 1e-2, 1e-8)
        valid = window_sum > floor
        audio[valid] /= window_sum[valid]
        audio[~valid] = 0.0

        return audio

    def _token_budget(self) -> int:
        """Max tokens per segment the *acoustic* model can render.

        The vocoder no longer constrains this — run_vocos() chunks whatever it
        is given (see _run_vocos_frames).  What remains is the acoustic model:
        the RKNN buckets are compiled at a fixed x_len (MAX_SEQ_LEN), while the
        ORT path takes dynamic shapes and has no such limit.  Enforcing the
        RKNN bucket width on ORT was silently dropping tokens off the end of
        long segments for no reason.
        """
        if getattr(self, '_matcha_backend', None) == 'ort':
            return int(os.environ.get('MATCHA_ORT_MAX_PHONEMES', '256'))
        return MAX_SEQ_LEN

    def _split_by_budget(self, seg: str, budget: int) -> list[str]:
        """Hard-split a segment that has no usable punctuation left.

        Breaks at word boundaries for latin text and at character boundaries
        for CJK, accumulating until the token budget is reached.  Without this
        a long unpunctuated clause fell through to the ``tokens[:MAX_SEQ_LEN]``
        truncation and simply lost its tail.
        """
        import re
        atoms = re.findall(r'[一-鿿]|\S+\s*', seg)
        if not atoms:
            return [seg]

        out: list[str] = []
        cur = ''
        for atom in atoms:
            cand = cur + atom
            if cur and len(self.text_to_tokens(cand)) > budget:
                out.append(cur.strip())
                cur = atom
            else:
                cur = cand
        if cur.strip():
            out.append(cur.strip())
        return out or [seg]

    def _split_text(self, text: str, speed: float = 1.0) -> list[str]:
        """将文本按句子分割，确保每段不超过声学模型容量。

        Three-tier cascade: sentence-end punctuation -> soft punctuation ->
        hard token-budget split.
        """
        import re
        budget = self._token_budget()

        # 按句末标点分割。
        #
        # sherpa also cuts a chunk at commas, but we deliberately do not:
        # punctuation now reaches the acoustic model as a token, so the pause
        # is rendered by the model itself, and cutting again would splice two
        # independently-generated clips together without sherpa's
        # inter-sentence silence.  Measured: comma-chunking left English at
        # 0.0% WER but pushed Chinese from 0.0% to 3.2% CER -- "空调调到"
        # came out as "空调跳到" in 5/5 runs, the tone smeared across the
        # splice.  Cutting only at sentence ends keeps both at 0.
        #
        # ASCII '.' needs a lookahead so decimals ("26.5") are not split;
        # without it English statements -- which almost always end in '.' --
        # were never sentence-split at all.
        sentence_end = r'([。！？；;]|[!?.](?=\s|$))'
        segments = re.split(sentence_end, text)
        # 将标点重新附加到前一段
        result = []
        for i in range(0, len(segments), 2):
            seg = segments[i]
            if i + 1 < len(segments):
                seg += segments[i + 1]
            seg = seg.strip()
            if seg:
                result.append(seg)
        if not result:
            return [text]

        # 对仍超出预算的段，按逗号等软标点进一步拆分
        final = []
        for seg in result:
            if len(self.text_to_tokens(seg)) <= budget:
                final.append(seg)
                continue

            sub_segs = re.split(r'([，,、：:])', seg)
            sub_result = []
            for j in range(0, len(sub_segs), 2):
                s = sub_segs[j]
                if j + 1 < len(sub_segs):
                    s += sub_segs[j + 1]
                s = s.strip()
                if s:
                    sub_result.append(s)

            # 软标点也救不了的，按 token 预算硬切
            for s in (sub_result or [seg]):
                if len(self.text_to_tokens(s)) <= budget:
                    final.append(s)
                else:
                    final.extend(self._split_by_budget(s, budget))
        return final

    @staticmethod
    def _smooth_mel(mel: np.ndarray) -> np.ndarray:
        """Fix FP16 energy anomalies in RKNN mel via adaptive per-frame correction.

        RKNN FP16 estimator produces localized energy dips (40% of expected) and
        spikes (200%) at individual frames. Instead of blanket smoothing (which
        blurs good frames), we detect anomalous frames by comparing per-frame
        energy against a local median, then blend only those frames with their
        neighbors.

        Args:
            mel: [1, 80, T] mel spectrogram

        Returns:
            Corrected mel with same shape.
        """
        m = mel[0]  # [80, T]
        T = m.shape[1]
        if T < 5:
            return mel

        # Per-frame energy
        energy = np.mean(m ** 2, axis=0)  # [T]

        # Local median energy (window=5)
        pad = 2
        e_padded = np.pad(energy, pad, mode='reflect')
        # Sliding window median via sorted approach
        local_med = np.array([
            np.median(e_padded[i:i + 5]) for i in range(T)
        ])

        # Detect anomalous frames: energy ratio vs local median
        ratio = energy / (local_med + 1e-8)
        # Anomaly = frame where energy < 50% or > 180% of local median
        anomaly = (ratio < 0.5) | (ratio > 1.8)
        n_anomaly = np.sum(anomaly)
        if n_anomaly == 0:
            return mel

        # Blend anomalous frames with average of their 2 neighbors
        result = m.copy()
        for t in range(T):
            if anomaly[t]:
                left = max(0, t - 1)
                right = min(T - 1, t + 1)
                if left == t:
                    result[:, t] = m[:, right]
                elif right == t:
                    result[:, t] = m[:, left]
                else:
                    result[:, t] = (m[:, left] + m[:, right]) * 0.5

        return result[np.newaxis]

    def _synthesize_segment(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """合成单个文本段（不超过 MAX_SEQ_LEN 音素）。"""
        metadata = {}

        # Step 1: 文本 → tokens
        t0 = time.perf_counter()
        tokens = self.text_to_tokens(text)
        metadata['text_frontend_ms'] = (time.perf_counter() - t0) * 1000
        metadata['num_tokens'] = len(tokens)

        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32), metadata

        # 截断超长 tokens（_split_text 之后应当不再触发；触发即为漏网，必须留痕）
        budget = self._token_budget()
        if len(tokens) > budget:
            import logging
            logging.getLogger(__name__).warning(
                "segment exceeded token budget (%d > %d, backend=%s) after "
                "splitting — dropping %d tokens from %r",
                len(tokens), budget, self._matcha_backend,
                len(tokens) - budget, text[:60],
            )
            tokens = tokens[:budget]

        # Step 2: Matcha RKNN
        t0 = time.perf_counter()
        mel, mel_frames = self.run_matcha(tokens, noise_scale, 1.0 / speed)
        metadata['matcha_ms'] = (time.perf_counter() - t0) * 1000

        # Step 3: Vocos RKNN
        t0 = time.perf_counter()
        audio = self.run_vocos(mel, mel_frames)
        metadata['vocos_ms'] = (time.perf_counter() - t0) * 1000

        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
    ) -> tuple[np.ndarray, dict]:
        """
        合成语音，自动分句处理超长文本

        Args:
            text: 输入文本 (中文)
            speed: 语速 (1.0 = 正常)
            noise_scale: 噪声强度

        Returns:
            audio: 音频样本 (float32, [-1, 1])
            metadata: 元数据 (耗时等)
        """
        segments = self._split_text(text, speed)
        all_audio = []
        total_text_frontend_ms = 0.0
        total_matcha_ms = 0.0
        total_vocos_ms = 0.0
        total_num_tokens = 0

        for seg in segments:
            audio_seg, meta_seg = self._synthesize_segment(seg, speed, noise_scale)
            if len(audio_seg) > 0:
                all_audio.append(audio_seg)
            total_text_frontend_ms += meta_seg.get('text_frontend_ms', 0.0)
            total_matcha_ms += meta_seg.get('matcha_ms', 0.0)
            total_vocos_ms += meta_seg.get('vocos_ms', 0.0)
            total_num_tokens += meta_seg.get('num_tokens', 0)

        audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)

        gain, clip = utterance_gain(audio)
        if gain != 1.0:
            audio = audio * gain
            if clip:
                audio = np.clip(audio, -1.0, 1.0)

        metadata = {
            'num_tokens': total_num_tokens,
            'text_frontend_ms': total_text_frontend_ms,
            'matcha_ms': total_matcha_ms,
            'vocos_ms': total_vocos_ms,
        }
        metadata['duration_s'] = len(audio) / SAMPLE_RATE
        metadata['total_ms'] = sum(v for k, v in metadata.items() if k.endswith('_ms'))
        if metadata['duration_s'] > 0:
            metadata['rtf'] = metadata['total_ms'] / 1000 / metadata['duration_s']

        return audio.astype(np.float32), metadata


def create_rknn_tts_backend(model_dir: str = None) -> RKNNMatchaVocoder:
    """
    创建 RKNN TTS 后端

    Args:
        model_dir: 模型目录，默认从环境变量获取
    """
    if model_dir is None:
        model_dir = os.environ.get('TTS_MODEL_DIR', '/home/cat/models')

    model_dir = Path(model_dir)

    matcha_name = os.environ.get('MATCHA_MODEL', 'matcha-s64.rknn')
    vocos_name = os.environ.get('VOCOS_MODEL', 'vocos-16khz-600.rknn')

    return RKNNMatchaVocoder(
        matcha_rknn_path=str(model_dir / matcha_name),
        vocos_rknn_path=str(model_dir / vocos_name),
        lexicon_path=str(model_dir / 'matcha-icefall-zh-en' / 'lexicon.txt'),
        tokens_path=str(model_dir / 'matcha-icefall-zh-en' / 'tokens.txt'),
        data_dir=str(model_dir / 'matcha-icefall-zh-en' / 'espeak-ng-data'),
    )


class MatchaRKNNBackend:
    """TTSBackend wrapper around RKNNMatchaVocoder.

    Select via TTS_BACKEND=matcha_rknn.

    Note: intentionally duck-typed (not inheriting TTSBackend) to avoid
    importing tts_backend at module level. The synthesize_stream() fallback
    is provided explicitly below.
    """

    def __init__(self) -> None:
        self._engine: Optional[RKNNMatchaVocoder] = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "matcha_rknn"

    def is_ready(self) -> bool:
        """Return True if the engine is loaded and ready."""
        return self._engine is not None and self._engine._matcha_backend is not None

    def preload(self) -> None:
        """Create and load RKNNMatchaVocoder. Called once at startup."""
        self._engine = create_rknn_tts_backend()
        self._engine.load()

    def get_sample_rate(self) -> int:
        """Return audio sample rate in Hz."""
        return SAMPLE_RATE

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes.

        Args:
            text: Input text (Chinese / mixed Chinese-English).
            speaker_id: Ignored (Matcha model has a single speaker).
            speed: Speech rate multiplier (1.0 = normal). Defaults to 1.0.
            pitch_shift: Ignored (not supported by this backend).
            **kwargs: Forwarded to engine.synthesize() (e.g. noise_scale).

        Returns:
            wav_bytes: PCM audio encoded as a WAV file.
            metadata: Dict with keys ``duration``, ``inference_time``, ``rtf``
                      plus per-stage timing from the engine.
        """
        import io
        import soundfile as sf

        if self._engine is None:
            raise RuntimeError("MatchaRKNNBackend.preload() has not been called")

        t_start = time.perf_counter()
        audio, engine_meta = self._engine.synthesize(
            text,
            speed=speed if speed is not None else 1.0,
            **{k: v for k, v in kwargs.items() if k in ("noise_scale",)},
        )
        inference_time = time.perf_counter() - t_start

        # Guard: if no audio was produced (e.g. text yielded no phonemes),
        # return a short silence instead of crashing downstream.
        if len(audio) == 0:
            import logging
            logging.getLogger(__name__).warning(
                "No audio produced for text: %r — returning 0.1s silence", text
            )
            audio = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)

        # Encode float32 audio → WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration = engine_meta.get("duration_s", len(audio) / SAMPLE_RATE)
        rtf = engine_meta.get("rtf", inference_time / duration if duration > 0 else 0.0)

        metadata = {
            "duration": duration,
            "inference_time": inference_time,
            "rtf": rtf,
            **engine_meta,
        }
        return wav_bytes, metadata

    supports_streaming: bool = True

    def synthesize_stream(self, text, speaker_id=0, speed=None, pitch_shift=None, **kwargs):
        """Yield (audio_float32_chunk, metadata) per sentence chunk.

        Splits text into sentences via _split_text and synthesizes each one
        independently, yielding each chunk as soon as it's ready. TTFA equals
        the synthesis latency of the first sentence (~100-200ms on RK3588).
        """
        engine: RKNNMatchaVocoder = self._engine
        _speed = float(speed) if speed is not None else 1.0
        _noise = float(kwargs.get("noise_scale", 0.667))

        # Level has to match synthesize(): the batch path normalizes, and this
        # path used to reach it via a non-streaming fallback.  Yielding raw
        # segments here would make the streaming path (which is what the v2v
        # conversation loop uses) quieter than /tts for the same text.
        #
        # The gain is fixed from the first segment and reused: normalizing each
        # segment on its own would pump the level between sentences of one
        # reply, since a segment containing a plosive gets pushed down while a
        # quiet one gets pushed up.
        gain: float | None = None
        clip = False

        sentences = engine._split_text(text, _speed)
        for seg in sentences:
            audio_seg, seg_meta = engine._synthesize_segment(seg, _speed, _noise)
            if len(audio_seg) == 0:
                continue
            if gain is None:
                gain, clip = utterance_gain(audio_seg)
            if gain != 1.0:
                audio_seg = audio_seg * gain
                if clip:
                    audio_seg = np.clip(audio_seg, -1.0, 1.0)
            yield audio_seg.astype("float32"), seg_meta

    def cleanup(self) -> None:
        """Release RKNN resources."""
        if self._engine is not None:
            self._engine.release()
            self._engine = None


# 命令行测试
if __name__ == '__main__':
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description='RKNN TTS 测试')
    parser.add_argument('--text', '-t', default='你好世界', help='输入文本')
    parser.add_argument('--output', '-o', default='/tmp/rknn_tts.wav', help='输出文件')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='语速')
    args = parser.parse_args()

    print(f"输入: {args.text}")
    print("\n加载模型...")

    engine = create_rknn_tts_backend()
    engine.load()
    print("模型加载完成")

    print("\n合成中...")
    audio, meta = engine.synthesize(args.text, speed=args.speed)

    print(f"\n结果:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    if len(audio) > 0:
        sf.write(args.output, audio, SAMPLE_RATE)
        print(f"\n保存: {args.output}")

    engine.release()