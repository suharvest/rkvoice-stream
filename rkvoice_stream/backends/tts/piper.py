"""Piper VITS TTS Backend for RKNN NPU.

Multi-language support: one RKNN model per language, auto-detect or manual switch.
Select via TTS_BACKEND=piper_rknn.

Env vars:
  PIPER_MODEL_DIR: directory with language subdirs (default: /opt/piper-models)
  PIPER_LANGUAGES: comma-separated languages to preload (default: en_US)
  PIPER_DEFAULT_LANG: default language (default: en_US)
  PIPER_SEQ_LEN: max phoneme sequence length (default: 128)
  KOKORO_VOICE: default Kokoro voice for Japanese (default: jf_alpha)

Model directory structure (hybrid mode, preferred):
  /opt/piper-models/{lang}/encoder.onnx       -- Encoder+DP+LR on ORT CPU
  /opt/piper-models/{lang}/flow_decoder.rknn   -- Flow+Decoder on RKNN NPU
  /opt/piper-models/{lang}/model.onnx.json     -- Piper phoneme config
  (or config.json)

Model directory structure (legacy full-RKNN mode, fallback):
  /opt/piper-models/{lang}/model.rknn       -- Full model on RKNN NPU
  /opt/piper-models/{lang}/model.onnx.json  -- Piper phoneme config

Japanese CPU fallback (sherpa-onnx Kokoro v1.0):
  /opt/piper-models/ja_JP/kokoro-v1.0.onnx
  /opt/piper-models/ja_JP/kokoro-v1.0-tokens.txt
  /opt/piper-models/ja_JP/voices.bin
  (download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
SEQ_LEN = int(os.environ.get("PIPER_SEQ_LEN", "128"))
MODEL_DIR = os.environ.get("PIPER_MODEL_DIR", "/opt/piper-models")
DEFAULT_LANG = os.environ.get("PIPER_DEFAULT_LANG", "en_US")
PRELOAD_LANGS = [
    lang.strip()
    for lang in os.environ.get("PIPER_LANGUAGES", "en_US").split(",")
    if lang.strip()
]

# Japanese CPU fallback settings
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "jf_alpha")
# Valid Kokoro Japanese voices: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
_KOKORO_VOICES = {"jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"}
_JA_LANGS = {"ja", "ja_JP"}

# RMS silence-trim threshold (float32 scale)
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_FRAME_SIZE = 512


# ---------------------------------------------------------------------------
# Language detection (Unicode-range heuristics)
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect language from text using Unicode ranges.

    Returns a Piper language code like 'en_US', 'zh_CN', 'ja_JP', etc.
    Falls back to DEFAULT_LANG if unknown.
    """
    cjk = 0
    hiragana_katakana = 0
    hangul = 0
    cyrillic = 0
    latin = 0
    arabic = 0
    devanagari = 0

    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF:
            cjk += 1
        elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            hiragana_katakana += 1
        elif 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif 0x0900 <= cp <= 0x097F:
            devanagari += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            latin += 1

    total = len(text) or 1
    scores = {
        "zh_CN": cjk / total,
        "ja_JP": hiragana_katakana / total,
        "ko_KR": hangul / total,
        "ru_RU": cyrillic / total,
        "ar_AR": arabic / total,
        "hi_IN": devanagari / total,
        "en_US": latin / total,
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] < 0.05:
        return DEFAULT_LANG
    return best


# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------

try:
    from piper_phonemize import phonemize_espeak as _phonemize_espeak_lib
    _HAS_PIPER_PHONEMIZE = True
    logger.info("piper_phonemize available — using native phonemizer")
except ImportError:
    _HAS_PIPER_PHONEMIZE = False
    logger.debug("piper_phonemize not available — falling back to espeak-ng subprocess")


# Clause terminators. Piper voices are trained on phoneme sequences that carry
# these as tokens (phoneme_id_map has ids for them) -- the token is what makes
# the model render the pause and the falling/rising contour. `espeak-ng --ipa`
# drops them: it prints one bare line per clause and nothing else (measured on
# the speech image, espeak-ng 1.51: "Hello, world." -> "həlˈoʊ\nwˈɜːld"). The
# reference frontend (piper_phonemize) re-attaches the terminator after each
# clause; _phonemize_subprocess does the same.
_PUNCT_NORMALIZE = {
    '，': ',', '、': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
}
# An ASCII terminator only ends a clause when whitespace, a closing quote or
# bracket, or the end of the text follows, so "1,000", "10:30" and "3.5" stay
# whole -- the same call espeak makes. Full-width marks need no lookahead.
_CLAUSE_END_RE = re.compile(
    r'([,.;:!?]+)(?=[\s"\'”’)\]]|$)|([，、。！？；：]+)'
)


def _split_clauses(text: str) -> list[tuple[str, str]]:
    """Split text into (clause, terminator) pairs; terminator may be ''."""
    clauses: list[tuple[str, str]] = []
    pos = 0
    for m in _CLAUSE_END_RE.finditer(text):
        # A run ("?!", "...") collapses onto its first mark.
        mark = (m.group(1) or m.group(2))[0]
        clauses.append((text[pos:m.start()].strip(), _PUNCT_NORMALIZE.get(mark, mark)))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        clauses.append((tail, ""))
    return [(c, p) for c, p in clauses if c]


def _phonemize_subprocess(text: str, voice: str) -> str:
    """Phonemize via the espeak-ng CLI, keeping the clause terminators."""
    clauses = _split_clauses(text)
    if not clauses:
        return ""
    # One espeak call for the whole text, one clause per input line. espeak may
    # still break a clause further (it has its own clause rules), and then the
    # lines no longer pair up with ours -- fall back to one call per clause.
    lines = [
        ln.strip()
        for ln in _run_espeak("\n".join(c for c, _ in clauses), voice).splitlines()
        if ln.strip()
    ]
    if len(lines) != len(clauses):
        lines = [
            " ".join(_run_espeak(c, voice).split()) for c, _ in clauses
        ]
    return " ".join(
        f"{ipa}{mark}" for ipa, (_, mark) in zip(lines, clauses) if ipa
    )


def _run_espeak(text: str, voice: str) -> str:
    """Call espeak-ng via subprocess to get IPA phonemes."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-v", voice, "-q", "--", text],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                "espeak-ng returned %d for voice=%s: %s",
                result.returncode, voice, result.stderr.strip(),
            )
        return result.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError(
            "espeak-ng not found. Install it: apt-get install espeak-ng"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng timed out")


def text_to_phonemes(text: str, voice: str) -> str:
    """Convert text to IPA phoneme string using piper_phonemize or espeak-ng."""
    if _HAS_PIPER_PHONEMIZE:
        # piper_phonemize returns list-of-list; flatten to string
        try:
            result = _phonemize_espeak_lib(text, voice)
            if isinstance(result, list):
                # One list of single-codepoint phonemes per sentence, word
                # separators (" ") and terminators included. Joining a
                # sentence with "" keeps both; joining with " " turned every
                # phoneme into its own word.
                return " ".join(
                    "".join(item) if isinstance(item, list) else str(item)
                    for item in result
                )
            return str(result)
        except Exception as exc:
            logger.warning("piper_phonemize failed (%s), falling back to subprocess", exc)

    return _phonemize_subprocess(text, voice)


def phonemes_to_ids(phoneme_str: str, phoneme_id_map: dict) -> list[int]:
    """Map IPA phoneme string to token IDs using the model's phoneme_id_map.

    Piper's phoneme_id_map uses special tokens:
      "^" = BOS, "$" = EOS, " " = word-separator, "_" = padding/blank

    VITS requires blank tokens (ID=0, "_") interspersed between every phoneme
    for the monotonic alignment search to work correctly.
    """
    # Remove zero-width joiner (espeak-ng 1.52+ uses U+200D in diphthongs)
    phoneme_str = phoneme_str.replace("\u200d", "")

    pad_id = phoneme_id_map.get("_", [0])[0]
    ids: list[int] = [pad_id]

    # BOS
    if "^" in phoneme_id_map:
        ids.extend(phoneme_id_map["^"])
        ids.append(pad_id)

    # Split on whitespace into words. The separator goes BETWEEN words only, as
    # in the reference id sequence: a terminator rides on the word before it
    # ("wˈɜːld." -> ... d . $), and nothing sits between the last phoneme and
    # EOS. Emitting it after every word put a " " before EOS that the voices
    # never saw in training.
    for i, token in enumerate(phoneme_str.split()):
        if i > 0 and " " in phoneme_id_map:
            ids.extend(phoneme_id_map[" "])
            ids.append(pad_id)
        if token in phoneme_id_map:
            ids.extend(phoneme_id_map[token])
            ids.append(pad_id)
        else:
            for ch in token:
                if ch in phoneme_id_map:
                    ids.extend(phoneme_id_map[ch])
                    ids.append(pad_id)

    # EOS
    if "$" in phoneme_id_map:
        ids.extend(phoneme_id_map["$"])
        ids.append(pad_id)

    return ids


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------

def _trim_silence(audio: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> np.ndarray:
    """Trim leading and trailing silence below RMS threshold."""
    if len(audio) == 0:
        return audio

    frame_size = SILENCE_FRAME_SIZE
    n_frames = len(audio) // frame_size

    if n_frames == 0:
        return audio

    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    nonsilent = np.where(rms > threshold)[0]
    if len(nonsilent) == 0:
        return audio

    start = nonsilent[0] * frame_size
    end = (nonsilent[-1] + 1) * frame_size
    return audio[start:end]


# Pause appended after a segment, by its final mark. _trim_silence removes the
# silence the model rendered for that mark, and segments are played back to
# back -- within one synthesize() call and across the per-sentence calls a
# dialogue server makes -- so without this sentences run into each other.
# Marks INSIDE a segment need nothing here: they reach the model as tokens.
_SENTENCE_END = frozenset(".!?。！？")
_CLAUSE_END = frozenset(",;:，、；：")
_TRAILING_CLOSERS = " \t\n\"'”’)]"


def _segment_pause_ms(text: str) -> float:
    """Pause for the segment's final mark; 0 when it ends mid-phrase.

    Read per call, not at import: a module-level read goes stale across a
    profile hot-reload.
    """
    last = text.rstrip(_TRAILING_CLOSERS)[-1:]
    if last in _SENTENCE_END:
        name, default = "PIPER_SENTENCE_PAUSE_MS", 300.0
    elif last in _CLAUSE_END:
        name, default = "PIPER_CLAUSE_PAUSE_MS", 150.0
    else:
        return 0.0
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, os.environ.get(name))
        return default
    return value if math.isfinite(value) and value >= 0 else default


# ---------------------------------------------------------------------------
# Per-language model context
# ---------------------------------------------------------------------------

class _LangModel:
    """Holds model context + config for a single language.

    Supports two modes:
    - Hybrid (preferred): encoder.onnx on ORT CPU + flow_decoder.rknn on NPU.
    - Legacy: full model.rknn on NPU (fixed seq_len).

    Auto-detects mode based on available files in model_dir.
    """

    # Fixed mel length for NPU flow_decoder (must match RKNN build)
    MAX_MEL_LEN = 256
    # HiFi-GAN hop size: each mel frame = 256 audio samples
    HOP_SIZE = 256

    def __init__(self, lang: str, model_dir: Path) -> None:
        self.lang = lang
        self.model_dir = model_dir
        self._rknn = None
        self._encoder = None  # ORT session for hybrid mode
        self._hybrid = False
        self.config: dict = {}
        self.phoneme_id_map: dict = {}
        self.espeak_voice: str = "en-us"
        self.noise_scale: float = 0.667
        self.length_scale: float = 1.0
        self.noise_w: float = 0.8
        self.sample_rate: int = SAMPLE_RATE
        # Window sizes the LOADED artifacts actually use. Defaults are the
        # module/class constants; _load_hybrid replaces them with the shapes
        # baked into the encoder, which is where the truth lives for an export
        # produced by models/tts/piper/split_piper_vits.py (both halves of that
        # split are static). Hardcoding them cost us: three artifact sets in
        # circulation are compiled at 128, 150 and 256 mel frames, and two of
        # the three mismatches produce no error at all.
        self.seq_len: int = SEQ_LEN
        self.mel_len: int = self.MAX_MEL_LEN

    def _load_config(self) -> None:
        """Load phoneme config from model directory."""
        # Try multiple config file names
        for name in ("config.json", "model.onnx.json"):
            config_path = self.model_dir / name
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                break
        else:
            # Try any .onnx.json file
            json_files = list(self.model_dir.glob("*.onnx.json"))
            if json_files:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                raise FileNotFoundError(
                    f"No config file found in {self.model_dir}. "
                    "Expected config.json or model.onnx.json."
                )

        audio_cfg = self.config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", SAMPLE_RATE)

        espeak_cfg = self.config.get("espeak", {})
        self.espeak_voice = espeak_cfg.get("voice", "en-us")

        self.phoneme_id_map = self.config.get("phoneme_id_map", {})

        inference_cfg = self.config.get("inference", {})
        self.noise_scale = inference_cfg.get("noise_scale", 0.667)
        self.length_scale = inference_cfg.get("length_scale", 1.0)
        self.noise_w = inference_cfg.get("noise_w", 0.8)

    def load(self) -> None:
        self._load_config()

        encoder_path = self.model_dir / "encoder.onnx"
        fd_rknn_path = self.model_dir / "flow_decoder.rknn"
        legacy_rknn_path = self.model_dir / "model.rknn"

        if encoder_path.exists() and fd_rknn_path.exists():
            self._load_hybrid(encoder_path, fd_rknn_path)
        elif legacy_rknn_path.exists():
            self._load_legacy(legacy_rknn_path)
        else:
            raise FileNotFoundError(
                f"No model files found in {self.model_dir}. "
                "Expected encoder.onnx + flow_decoder.rknn (hybrid) "
                "or model.rknn (legacy)."
            )

    def _probe_decoder_window(self) -> int:
        """Ask the decoder how many mel frames it was compiled for.

        There is no rknn-lite API for tensor shapes, and the two failure modes
        differ by runtime version: a decoder may reject a wrong length outright
        (inference() returns None) or accept it and return its own frame count
        anyway. Both are usable here — feed candidates until one returns, then
        take the window from the OUTPUT size, which is the model's own answer
        rather than what we guessed.
        """
        env = os.environ.get("PIPER_MEL_LEN", "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)

        candidates = [self.mel_len, 512, 256, 150, 128, 1024]
        seen = []
        for cand in dict.fromkeys(candidates):
            out = self._rknn.inference(inputs=[
                np.zeros((1, 192, cand), dtype=np.float32),
                np.zeros((1, 1, cand), dtype=np.float32),
            ])
            if not out or not len(out):
                seen.append(cand)
                continue
            samples = int(out[0].size)
            # A window is a whole number of hops by construction. Anything else
            # means the output is not what this code thinks it is, and guessing
            # from it would set a window that silently mis-slices every decode.
            if samples <= 0 or samples % self.HOP_SIZE:
                raise RuntimeError(
                    f"Piper {self.lang}: decoder returned {samples} samples for a "
                    f"{cand}-frame probe, which is not a whole number of "
                    f"{self.HOP_SIZE}-sample hops. The artifacts do not match what "
                    f"this backend expects."
                )
            return samples // self.HOP_SIZE
        raise RuntimeError(
            f"Piper {self.lang}: the decoder rejected every probed mel window "
            f"({seen}). Set PIPER_MEL_LEN to the value it was exported with."
        )

    def _load_hybrid(self, encoder_path: Path, fd_rknn_path: Path) -> None:
        """Load hybrid mode: encoder ORT CPU + flow_decoder RKNN NPU."""
        import onnxruntime as ort
        from rknnlite.api import RKNNLite

        self._encoder = ort.InferenceSession(
            str(encoder_path), providers=["CPUExecutionProvider"]
        )

        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(str(fd_rknn_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN for {self.lang}: ret={ret}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime for {self.lang}: ret={ret}")

        # Read the compiled window out of the encoder when it is static.
        # A dynamic encoder (upstream Piper ONNX, not our split) leaves the
        # defaults in place and says so, because then only the decoder knows.
        enc_in = {i.name: i.shape for i in self._encoder.get_inputs()}
        enc_out = [o.shape for o in self._encoder.get_outputs()]
        seq_dim = (enc_in.get("input") or [None, None])[-1]
        mel_dim = enc_out[0][-1] if enc_out and len(enc_out[0]) == 3 else None
        if isinstance(seq_dim, int):
            self.seq_len = seq_dim
        if isinstance(mel_dim, int):
            self.mel_len = mel_dim
            source = "encoder graph"
        else:
            # A dynamic encoder is the normal shape for an export where only
            # the decoder half is frozen (split_piper_vits.py leaves the
            # encoder dynamic; some older exports froze both). The decoder is
            # then the only thing that knows the window, so ask it.
            self.mel_len = self._probe_decoder_window()
            source = "decoder probe"
        logger.info(
            "Piper %s: mel window %d frames (from %s)",
            self.lang, self.mel_len, source,
        )

        # Prove the decoder agrees, with one zero-input decode. rknn-lite does
        # NOT reliably reject a wrong length: measured 2026-09-20, a 128-frame
        # decoder accepted 150/256/512 inputs without error and returned its own
        # 128 frames every time (silently truncated audio), while a 150-frame
        # one rejected 128 with RKNN_ERR_PARAM_INVALID and inference() returned
        # None, which this backend then turned into empty audio. Neither is
        # visible without checking, so check.
        probe = self._rknn.inference(inputs=[
            np.zeros((1, 192, self.mel_len), dtype=np.float32),
            np.zeros((1, 1, self.mel_len), dtype=np.float32),
        ])
        if probe is None or len(probe) == 0:
            raise RuntimeError(
                f"Piper {self.lang}: decoder rejected a {self.mel_len}-frame "
                f"input ({fd_rknn_path}). The encoder and the decoder were "
                f"compiled for different windows; re-export both with the same "
                f"--mel-len (models/tts/piper/split_piper_vits.py)."
            )
        got_frames = probe[0].size // self.HOP_SIZE
        if got_frames != self.mel_len:
            raise RuntimeError(
                f"Piper {self.lang}: decoder returned {got_frames} frames for a "
                f"{self.mel_len}-frame input ({fd_rknn_path}) — it was compiled "
                f"for a different window, and rknn-lite did not report it. "
                f"Audio would be silently truncated to "
                f"{got_frames * self.HOP_SIZE / self.sample_rate:.2f}s."
            )

        self._hybrid = True
        logger.info(
            "Loaded Piper HYBRID model for %s (voice=%s, sr=%d, seq_len=%d, "
            "mel_len=%d → %.2fs max per inference) encoder=ORT_CPU, "
            "decoder=RKNN_NPU",
            self.lang, self.espeak_voice, self.sample_rate, self.seq_len,
            self.mel_len, self.mel_len * self.HOP_SIZE / self.sample_rate,
        )

    def _load_legacy(self, rknn_path: Path) -> None:
        """Load legacy full-RKNN mode."""
        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN for {self.lang}: ret={ret}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime for {self.lang}: ret={ret}")

        self._hybrid = False
        logger.info(
            "Loaded Piper LEGACY RKNN model for %s (voice=%s, sr=%d)",
            self.lang, self.espeak_voice, self.sample_rate,
        )

    def release(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
            self._rknn = None
        self._encoder = None

    def infer(
        self,
        token_ids: list[int],
        length_scale: float,
        noise_scale: float,
        noise_w: float,
    ) -> np.ndarray:
        """Run inference. Returns raw float32 audio samples."""
        if self._hybrid:
            return self._infer_hybrid(token_ids, length_scale, noise_scale, noise_w)
        return self._infer_legacy(token_ids, length_scale, noise_scale, noise_w)

    def _infer_hybrid(
        self,
        token_ids: list[int],
        length_scale: float,
        noise_scale: float,
        noise_w: float,
    ) -> np.ndarray:
        """Hybrid inference: encoder on CPU, flow+decoder on NPU."""
        n = min(len(token_ids), self.seq_len)

        # Pad tokens to the encoder's compiled phoneme length
        tokens = np.zeros((1, self.seq_len), dtype=np.int64)
        tokens[0, :n] = token_ids[:n]
        lengths = np.array([n], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        # Step 1: Encoder on CPU
        enc_inputs: dict[str, np.ndarray] = {
            "input": tokens,
            "input_lengths": lengths,
            "scales": scales,
        }
        # Add sid if encoder expects it
        enc_input_names = {inp.name for inp in self._encoder.get_inputs()}
        if "sid" in enc_input_names:
            enc_inputs["sid"] = np.array([0], dtype=np.int64)
        # Add x_mask if encoder expects it (hoisted from internal in fixed ONNX)
        if "x_mask" in enc_input_names:
            x_mask = np.zeros((1, 1, self.seq_len), dtype=np.float32)
            x_mask[0, 0, :n] = 1.0
            enc_inputs["x_mask"] = x_mask
        # Add audio_length if encoder expects it
        if "audio_length" in enc_input_names:
            enc_inputs["audio_length"] = np.array([0], dtype=np.int64)
        # Add cumulative_durations if encoder expects it
        if "cumulative_durations" in enc_input_names:
            enc_inputs["cumulative_durations"] = np.zeros(
                (self.seq_len, 1), dtype=np.float32
            )

        enc_out = self._encoder.run(None, enc_inputs)
        z = enc_out[0]        # (1, 192, mel_len)
        y_mask = enc_out[1]   # (1, 1, mel_len)
        mel_len = z.shape[2]

        # Step 2: Pad to fixed size for NPU
        # Step 3: Flow+Decoder on NPU, in as many windows as the mel needs
        return self._decode_mel(z, y_mask, mel_len)

    def _decode_mel(
        self,
        z: np.ndarray,
        y_mask: np.ndarray,
        total_frames: int,
    ) -> np.ndarray:
        """Render `total_frames` mel frames through the fixed-window decoder.

        Mirrors the window schedule matcha.py uses for its equally fixed-window
        vocoder (_run_vocos_frames): each chunk carries `ctx` frames of context
        on both sides which are then discarded, so the convolutional receptive
        field is fed properly across the seams.

        It stitches AUDIO rather than spectra, which is the one place it has to
        differ: matcha concatenates (mag, cos, sin) and runs a single ISTFT at
        the end, and this decoder is a HiFi-GAN that emits samples directly.
        Discarding context is what keeps the seams clean here.

        Before this, anything past one window was dropped without a trace, and
        one window is only a few seconds — ordinary sentences hit it.
        """
        cap = self.mel_len
        if cap <= 0:
            # Everything below divides the work into windows of this size; a
            # zero would make the loop advance by nothing and spin forever.
            raise RuntimeError(
                f"Piper {self.lang}: decoder window is {cap}, which cannot be "
                f"used. Set PIPER_MEL_LEN to the exported value."
            )

        def _emit(w_start: int, w_end: int, keep_lo: int, keep_hi: int) -> np.ndarray:
            z_pad = np.zeros((1, 192, cap), dtype=np.float32)
            mask_pad = np.zeros((1, 1, cap), dtype=np.float32)
            n = w_end - w_start
            z_pad[:, :, :n] = z[:, :, w_start:w_end]
            mask_pad[:, :, :n] = y_mask[:, :, w_start:w_end]

            out = self._rknn.inference(inputs=[z_pad, mask_pad])
            if out is None or len(out) == 0:
                # Not recoverable and not silent: returning an empty array here
                # made a shape mismatch look like a successful synthesis of
                # nothing.
                raise RuntimeError(
                    f"Piper {self.lang}: decoder inference returned nothing for "
                    f"a {cap}-frame input. The artifacts loaded fine, so this is "
                    f"a runtime failure rather than a window mismatch (that is "
                    f"checked at load)."
                )
            audio = out[0].flatten()
            lo = (keep_lo - w_start) * self.HOP_SIZE
            hi = (keep_hi - w_start) * self.HOP_SIZE
            return audio[lo:hi]

        if total_frames <= cap:
            return _emit(0, total_frames, 0, total_frames).astype(np.float32)

        ctx = min(32, cap // 8)
        stride = cap - 2 * ctx
        if stride <= 0:  # pathologically small window
            ctx, stride = 0, cap

        parts = []
        pos = 0
        while pos < total_frames:
            w_end = min(total_frames, max(pos - ctx, 0) + cap)
            w_start = max(0, w_end - cap)
            keep_end = min(total_frames, pos + stride)
            parts.append(_emit(w_start, w_end, pos, keep_end))
            pos = keep_end

        logger.debug(
            "Piper %s: %d mel frames rendered in %d windows of %d (ctx=%d)",
            self.lang, total_frames, len(parts), cap, ctx,
        )
        return np.concatenate(parts).astype(np.float32)

    def _infer_legacy(
        self,
        token_ids: list[int],
        length_scale: float,
        noise_scale: float,
        noise_w: float,
    ) -> np.ndarray:
        """Legacy full-RKNN inference with fixed seq_len."""
        n = min(len(token_ids), SEQ_LEN)
        tokens = np.zeros((1, SEQ_LEN), dtype=np.int64)
        tokens[0, :n] = token_ids[:n]
        lengths = np.array([n], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        out = self._rknn.inference(inputs=[tokens, lengths, scales])
        if out is None or len(out) == 0:
            return np.zeros(0, dtype=np.float32)
        return out[0].flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Japanese CPU fallback — sherpa-onnx Kokoro v1.0
# ---------------------------------------------------------------------------

class _JaKokoroModel:
    """CPU-based Japanese TTS using sherpa-onnx Kokoro v1.0.

    Replaces the RKNN path for ja/ja_JP because the Kokoro vocoder output
    dimension (13081) exceeds RKNN NPU register limit (8191).

    Expected files under model_dir:
      kokoro-v1.0.onnx
      kokoro-v1.0-tokens.txt
      voices.bin
    """

    # Kokoro v1.0 outputs 24 kHz audio
    SAMPLE_RATE = 24000

    def __init__(self, lang: str, model_dir: Path, voice: str = KOKORO_VOICE) -> None:
        self.lang = lang
        self.model_dir = model_dir
        self.voice = voice if voice in _KOKORO_VOICES else KOKORO_VOICE
        self._tts = None
        self.sample_rate: int = self.SAMPLE_RATE
        # Expose same attrs _synthesize_segment reads from _LangModel
        self.espeak_voice: str = "ja"

    def load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError:
            raise RuntimeError(
                "sherpa-onnx not installed. Add 'sherpa-onnx' to requirements "
                "or run: pip install sherpa-onnx"
            )

        model_path = self.model_dir / "kokoro-v1.0.onnx"
        tokens_path = self.model_dir / "kokoro-v1.0-tokens.txt"
        voices_path = self.model_dir / "voices.bin"

        for p in (model_path, tokens_path, voices_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Kokoro model file not found: {p}\n"
                    "Download from https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models\n"
                    "  kokoro-multi-lang-v1_0.tar.bz2"
                )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(model_path),
                    voices=str(voices_path),
                    tokens=str(tokens_path),
                    # data_dir is optional; espeak-ng data auto-detected
                ),
                num_threads=2,
                debug=False,
                provider="cpu",
            ),
            rule_fsts="",
            max_num_sentences=1,
        )

        if not tts_config.validate():
            raise RuntimeError("Invalid sherpa-onnx Kokoro TTS config")

        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self._tts.sample_rate
        logger.info(
            "Loaded Kokoro v1.0 (CPU) for %s (voice=%s, sr=%d)",
            self.lang, self.voice, self.sample_rate,
        )

    def release(self) -> None:
        # sherpa-onnx manages its own lifetime; just drop the reference
        self._tts = None

    def infer(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text directly (no phoneme pre-processing needed).

        sherpa-onnx handles grapheme-to-phoneme internally for Japanese.
        Returns float32 audio samples at self.sample_rate.
        """
        if self._tts is None:
            raise RuntimeError("_JaKokoroModel not loaded")

        audio = self._tts.generate(text, sid=0, speed=speed)
        if audio.samples is None or len(audio.samples) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.array(audio.samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?;。！？；\n])\s*')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming synthesis."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class PiperRKNNBackend:
    """Piper VITS TTS backend using RKNN NPU.

    Intentionally duck-typed (not inheriting TTSBackend) to avoid circular
    imports — same pattern as MatchaRKNNBackend.

    Select via TTS_BACKEND=piper_rknn.
    """

    supports_streaming: bool = True

    def __init__(self) -> None:
        self._models: dict[str, "_LangModel | _JaKokoroModel"] = {}
        self._ready = False

    # ------------------------------------------------------------------
    # TTSBackend protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "piper_rknn"

    def is_ready(self) -> bool:
        return self._ready and bool(self._models)

    def get_sample_rate(self) -> int:
        # Return sample rate of default lang if loaded, else 22050
        if DEFAULT_LANG in self._models:
            return self._models[DEFAULT_LANG].sample_rate
        if self._models:
            return next(iter(self._models.values())).sample_rate
        return SAMPLE_RATE

    def preload(self) -> None:
        """Load RKNN models for all configured languages.

        Japanese (ja/ja_JP) is handled by sherpa-onnx Kokoro v1.0 on CPU
        instead of RKNN NPU (vocoder dimension exceeds NPU limit).
        """
        model_root = Path(MODEL_DIR)
        for lang in PRELOAD_LANGS:
            lang_dir = model_root / lang
            if not lang_dir.exists():
                logger.warning("Piper model dir not found for %s: %s", lang, lang_dir)
                continue
            try:
                if lang in _JA_LANGS or lang.split("_")[0] == "ja":
                    # Japanese: use CPU Kokoro fallback
                    m = _JaKokoroModel(lang, lang_dir)
                    m.load()
                else:
                    m = _LangModel(lang, lang_dir)
                    m.load()
                self._models[lang] = m
            except Exception as exc:
                logger.error("Failed to load model for %s: %s", lang, exc)

        if self._models:
            self._ready = True
            logger.info(
                "PiperRKNNBackend ready. Loaded languages: %s",
                list(self._models.keys()),
            )
        else:
            logger.error("PiperRKNNBackend: no models loaded — backend not ready")

    def cleanup(self) -> None:
        """Release all RKNN contexts."""
        for m in self._models.values():
            m.release()
        self._models.clear()
        self._ready = False

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def _get_model(self, lang: Optional[str]) -> "_LangModel | _JaKokoroModel":
        """Resolve language to a loaded model, with fallback."""
        if lang and lang in self._models:
            return self._models[lang]
        if lang:
            # Try prefix match: "zh" -> "zh_CN"
            prefix = lang.split("_")[0].lower()
            for key, m in self._models.items():
                if key.lower().startswith(prefix):
                    return m
            logger.warning("Language %r not loaded, falling back to default", lang)
        if DEFAULT_LANG in self._models:
            return self._models[DEFAULT_LANG]
        # Last resort: first available
        return next(iter(self._models.values()))

    def _synthesize_segment(
        self,
        text: str,
        lang_model,  # _LangModel | _JaKokoroModel
        speed: float = 1.0,
        noise_scale: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """Synthesize a single text segment. Returns (audio_float32, meta).

        Dispatches to CPU Kokoro path for Japanese, RKNN path for all others.
        """
        if isinstance(lang_model, _JaKokoroModel):
            return self._synthesize_segment_ja(text, lang_model, speed)

        meta: dict = {}

        t0 = time.perf_counter()
        try:
            phoneme_str = text_to_phonemes(text, lang_model.espeak_voice)
        except RuntimeError as exc:
            logger.error("Phonemization failed: %s", exc)
            return np.zeros(0, dtype=np.float32), {"error": str(exc)}
        meta["phonemize_ms"] = (time.perf_counter() - t0) * 1000

        token_ids = phonemes_to_ids(phoneme_str, lang_model.phoneme_id_map)
        meta["num_tokens"] = len(token_ids)

        if not token_ids:
            logger.warning("No token IDs for text %r (phonemes: %r)", text, phoneme_str)
            return np.zeros(0, dtype=np.float32), meta

        # Truncate to the phoneme length THIS model was built for. The module
        # constant is only the default; a static encoder carries its own.
        token_ids = token_ids[:getattr(lang_model, "seq_len", SEQ_LEN)]

        length_scale = lang_model.length_scale / max(speed, 0.1)
        ns = noise_scale if noise_scale is not None else lang_model.noise_scale

        t0 = time.perf_counter()
        audio = lang_model.infer(token_ids, length_scale, ns, lang_model.noise_w)
        meta["infer_ms"] = (time.perf_counter() - t0) * 1000

        audio = _trim_silence(audio)
        # The pause is speech timing, so it follows the requested speed.
        pause_ms = _segment_pause_ms(text) / max(speed, 0.1)
        if len(audio) > 0 and pause_ms > 0:
            pad = np.zeros(
                int(lang_model.sample_rate * pause_ms / 1000.0), dtype=audio.dtype
            )
            audio = np.concatenate([audio, pad])
        meta["duration_s"] = len(audio) / lang_model.sample_rate
        total_ms = meta["phonemize_ms"] + meta["infer_ms"]
        meta["total_ms"] = total_ms
        if meta["duration_s"] > 0:
            meta["rtf"] = total_ms / 1000.0 / meta["duration_s"]

        return audio, meta

    def _synthesize_segment_ja(
        self,
        text: str,
        lang_model: _JaKokoroModel,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, dict]:
        """Synthesize a Japanese segment via sherpa-onnx Kokoro on CPU."""
        meta: dict = {"phonemize_ms": 0.0}  # phonemization is internal to sherpa-onnx

        t0 = time.perf_counter()
        try:
            audio = lang_model.infer(text, speed=speed)
        except Exception as exc:
            logger.error("Kokoro inference failed: %s", exc)
            return np.zeros(0, dtype=np.float32), {"error": str(exc)}
        meta["infer_ms"] = (time.perf_counter() - t0) * 1000
        meta["num_tokens"] = len(text)  # character count as proxy

        audio = _trim_silence(audio)
        meta["duration_s"] = len(audio) / lang_model.sample_rate
        meta["total_ms"] = meta["infer_ms"]
        if meta["duration_s"] > 0:
            meta["rtf"] = meta["infer_ms"] / 1000.0 / meta["duration_s"]

        return audio, meta

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes.

        Args:
            text: Input text (any language).
            speaker_id: Ignored (single-speaker Piper models).
            speed: Speech rate multiplier (1.0 = normal).
            pitch_shift: Ignored (not supported).
            language: Language code ('en_US', 'zh_CN', …) or None for auto-detect.

        Returns:
            wav_bytes: PCM audio as WAV file.
            metadata: dict with duration, inference_time, rtf, etc.
        """
        import soundfile as sf

        if not self.is_ready():
            raise RuntimeError("PiperRKNNBackend.preload() has not been called")

        # Resolve language
        effective_lang = language or detect_language(text)
        lang_model = self._get_model(effective_lang)

        effective_speed = speed if speed is not None else 1.0
        noise_scale = kwargs.get("noise_scale", None)

        t_start = time.perf_counter()

        # Split into sentences for long texts
        sentences = _split_sentences(text)
        all_audio: list[np.ndarray] = []
        agg_meta: dict = {
            "phonemize_ms": 0.0,
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "num_tokens": 0,
        }

        for sentence in sentences:
            audio_seg, seg_meta = self._synthesize_segment(
                sentence, lang_model, effective_speed, noise_scale
            )
            if len(audio_seg) > 0:
                all_audio.append(audio_seg)
            for k in ("phonemize_ms", "infer_ms", "total_ms", "num_tokens"):
                agg_meta[k] = agg_meta.get(k, 0.0) + seg_meta.get(k, 0.0)

        audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)

        # Normalize
        if len(audio) > 0:
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.95

        inference_time = time.perf_counter() - t_start
        duration = len(audio) / lang_model.sample_rate
        rtf = inference_time / duration if duration > 0 else 0.0

        buf = io.BytesIO()
        sf.write(buf, audio, lang_model.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        metadata = {
            "duration": duration,
            "inference_time": inference_time,
            "rtf": rtf,
            "language": effective_lang,
            "espeak_voice": lang_model.espeak_voice,
            "backend": "kokoro_cpu" if isinstance(lang_model, _JaKokoroModel) else "piper_rknn",
            **agg_meta,
        }
        return wav_bytes, metadata

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """Stream TTS sentence-by-sentence.

        Yields (audio_float32_chunk, metadata) for each sentence.
        Allows the caller to start playing audio before the full text is done.
        """
        if not self.is_ready():
            raise RuntimeError("PiperRKNNBackend.preload() has not been called")

        effective_lang = language or detect_language(text)
        lang_model = self._get_model(effective_lang)
        effective_speed = speed if speed is not None else 1.0
        noise_scale = kwargs.get("noise_scale", None)

        sentences = _split_sentences(text)
        for sentence in sentences:
            audio_seg, seg_meta = self._synthesize_segment(
                sentence, lang_model, effective_speed, noise_scale
            )
            if len(audio_seg) == 0:
                continue

            peak = np.abs(audio_seg).max()
            if peak > 0:
                audio_seg = audio_seg / peak * 0.95

            duration = len(audio_seg) / lang_model.sample_rate
            meta = {
                "duration": duration,
                "language": effective_lang,
                **seg_meta,
            }
            yield audio_seg, meta


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Piper RKNN TTS smoke-test")
    parser.add_argument("--text", "-t", default="Hello, world.", help="Input text")
    parser.add_argument("--output", "-o", default="/tmp/piper_rknn_test.wav", help="Output WAV")
    parser.add_argument("--language", "-l", default=None, help="Language code (auto-detect if omitted)")
    parser.add_argument("--speed", "-s", type=float, default=1.0)
    args = parser.parse_args()

    backend = PiperRKNNBackend()
    backend.preload()

    if not backend.is_ready():
        print("Backend not ready — check model directory and logs")
        exit(1)

    wav_bytes, meta = backend.synthesize(args.text, speed=args.speed, language=args.language)
    with open(args.output, "wb") as f:
        f.write(wav_bytes)

    print(f"Saved {len(wav_bytes)} bytes to {args.output}")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    backend.cleanup()
