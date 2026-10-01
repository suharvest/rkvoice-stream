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
# A "." that belongs to the word before it, not to the end of anything. There is
# no lexicon here, so each rule is the narrowest that covers its common case:
#  - a title is always followed by a name;
#  - a dotted chain of single letters ("U.S.", "p.m.", "e.g.") is an abbreviation
#    -- "example.com" is not one -- unless a word
#    that plainly starts a sentence follows ("...in the U.S. They ship...");
#  - a single letter counts only as one of a run of initials ("J. K. Rowling").
#    On its own it is far more often the end of a sentence ("Plan B. Then...",
#    "vitamin C. It helps"), so "J. Smith" is the case given up.
# "etc." and "No." end real sentences too often to be listed.
_TITLES = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs"})
# Dotted abbreviations whose parts are not all single letters.
_DOTTED_ABBREVIATIONS = frozenset({"ph.d"})
_SENTENCE_STARTERS = frozenset({
    "The", "They", "We", "It", "He", "She", "I", "You", "This", "That", "These",
    "Those", "There", "Then", "But", "And", "So", "If", "When", "In", "On", "At",
    "A", "An", "Our", "My", "Your", "Please", "Do", "Does", "Is", "Are", "What",
})
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_INITIAL_BEFORE_RE = re.compile(r"(?:^|\s)[A-Za-z]\.\s+$")
_INITIAL_AFTER_RE = re.compile(r"\s+[A-Za-z]\.(?=\s|$)")
_NEXT_WORD_RE = re.compile(r"\s+([A-Za-z]+)")


def _is_abbreviation_dot(text: str, m: "re.Match") -> bool:
    """Whether the ASCII mark matched by ``m`` is a lone abbreviation period."""
    if m.group(1) != ".":
        return False
    before = text[:m.start()]
    word = _WORD_BEFORE_RE.search(before)
    if not word:
        return False
    w = word.group(1)
    if w.lower() in _TITLES:
        return True
    after = text[m.start() + 1:]
    if "." in w and (w.lower() in _DOTTED_ABBREVIATIONS
                     or all(len(part) == 1 for part in w.split("."))):
        # Single letters only: "U.S", "p.m", "e.g". "example.com" and "x.io"
        # are words with a dot in them, and the "." after one ends the sentence.
        nxt = _NEXT_WORD_RE.match(after)
        return not (nxt and nxt.group(1) in _SENTENCE_STARTERS)
    if len(w) == 1:
        return bool(_INITIAL_AFTER_RE.match(after)
                    or _INITIAL_BEFORE_RE.search(before[:word.start()]))
    return False


def _split_clauses(text: str) -> list[tuple[str, str]]:
    """Split text into (clause, terminator) pairs; terminator may be ''."""
    clauses: list[tuple[str, str]] = []
    pos = 0
    for m in _CLAUSE_END_RE.finditer(text):
        if _is_abbreviation_dot(text, m):
            continue
        # A run ("?!", "...") collapses onto its first mark.
        mark = (m.group(1) or m.group(2))[0]
        clauses.append((text[pos:m.start()].strip(), _PUNCT_NORMALIZE.get(mark, mark)))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        clauses.append((tail, ""))
    return [(c, p) for c, p in clauses if c]


def _phonemize_subprocess(text: str, voice: str,
                          espeak_cache: dict[tuple[str, str], str] | None = None) -> str:
    """Phonemize via the espeak-ng CLI, keeping the clause terminators."""
    clauses = _split_clauses(text)
    if not clauses:
        return ""
    # A single token-budget probe can reach the same clause through multiple
    # candidate prefixes. Keep this cache request-scoped: it removes duplicate
    # successful CLI calls without changing fallback or cross-request behavior.
    cache = espeak_cache if espeak_cache is not None else {}
    # One espeak call for the whole text, one clause per input line. espeak may
    # still break a clause further (it has its own clause rules), and then the
    # lines no longer pair up with ours -- fall back to one call per clause.
    lines = [
        ln.strip()
        for ln in _run_espeak("\n".join(c for c, _ in clauses), voice, cache=cache).splitlines()
        if ln.strip()
    ]
    if len(lines) != len(clauses):
        lines = [
            " ".join(_run_espeak(c, voice, cache=cache).split()) for c, _ in clauses
        ]
    return " ".join(
        f"{ipa}{mark}" for ipa, (_, mark) in zip(lines, clauses) if ipa
    )


def _run_espeak(text: str, voice: str,
                cache: dict[tuple[str, str], str] | None = None) -> str:
    """Call espeak-ng via subprocess to get IPA phonemes."""
    key = (voice, text)
    if cache is not None and key in cache:
        return cache[key]
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
        output = result.stdout.strip()
        # Do not cache failures or empty output: preserve the old retry
        # semantics for malformed input and transient CLI failures.
        if cache is not None and result.returncode == 0 and output:
            cache[key] = output
        return output
    except FileNotFoundError:
        raise RuntimeError(
            "espeak-ng not found. Install it: apt-get install espeak-ng"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng timed out")


def text_to_phonemes(text: str, voice: str,
                     espeak_cache: dict[tuple[str, str], str] | None = None) -> str:
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

    return _phonemize_subprocess(text, voice, espeak_cache=espeak_cache)


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
    # Include a final short frame.  Its RMS must use only the samples that
    # exist; padding it with zeros would dilute a voiced tail below threshold.
    n_full = len(audio) // frame_size
    rms_values: list[float] = []
    if n_full:
        frames = audio[: n_full * frame_size].reshape(n_full, frame_size)
        rms_values.extend(np.sqrt(np.mean(frames ** 2, axis=1)).tolist())
    remainder_start = n_full * frame_size
    if remainder_start < len(audio):
        partial = audio[remainder_start:]
        rms_values.append(float(np.sqrt(np.mean(partial ** 2))))

    nonsilent = np.where(np.asarray(rms_values) > threshold)[0]
    if len(nonsilent) == 0:
        return audio

    start = int(nonsilent[0]) * frame_size
    end = min((int(nonsilent[-1]) + 1) * frame_size, len(audio))
    return audio[start:end]


# Pause appended after a segment, by its final mark. _trim_silence removes the
# silence the model rendered for that mark, and segments are played back to
# back -- within one synthesize() call and across the per-sentence calls a
# dialogue server makes -- so without this sentences run into each other.
# Marks INSIDE a segment need nothing here: they reach the model as tokens.
_SENTENCE_END = frozenset(".!?。！？")
_CLAUSE_END = frozenset(",;:，、；：")
_TRAILING_CLOSERS = " \t\n\"'”’)]）」』"


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
        self._frontend_rknn = None
        self._encoder = None  # ORT session for hybrid mode
        self._remainder = None  # optional CPU remainder after frontend NPU
        self._frontend_npu = False
        self._frontend_manifest: dict = {}
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

        # The frontend split is opt-in because its remainder is an experimental
        # CPU graph and must never silently replace the production hybrid path.
        frontend_rknn = self.model_dir / "text_encoder.rknn"
        frontend_remainder = self.model_dir / "remainder.onnx"
        frontend_manifest = self.model_dir / "manifest.json"
        if os.environ.get("PIPER_ENABLE_FRONTEND_NPU", "0") == "1":
            if not (frontend_rknn.exists() and frontend_remainder.exists()
                    and frontend_manifest.exists() and (self.model_dir / "flow_decoder.rknn").exists()):
                raise FileNotFoundError(
                    f"Piper {self.lang}: frontend NPU enabled but requires "
                    "text_encoder.rknn, remainder.onnx, manifest.json, and flow_decoder.rknn"
                )
            self._load_frontend_npu(frontend_rknn, frontend_remainder, frontend_manifest)
            return

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

    def _load_frontend_npu(self, rknn_path: Path, remainder_path: Path,
                           manifest_path: Path) -> None:
        """Load an explicitly enabled text-encoder-NPU + CPU remainder pair."""
        import onnxruntime as ort
        from rknnlite.api import RKNNLite

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bucket = manifest.get("bucket", {}).get("input")
        if not isinstance(bucket, list) or len(bucket) != 2 or not all(isinstance(x, int) for x in bucket):
            raise RuntimeError(f"Piper {self.lang}: invalid frontend manifest bucket")
        if bucket[0] != 1 or bucket[1] <= 0:
            raise RuntimeError(f"Piper {self.lang}: unsupported frontend bucket {bucket}")
        self.seq_len = bucket[1]
        outputs = manifest.get("remainder", {}).get("outputs", [])
        shapes = manifest.get("remainder", {}).get("output_shapes", [])
        if (manifest.get("remainder", {}).get("output_semantic") != ["z", "y_mask"]
                or len(outputs) != 2 or len(shapes) != 2
                or len(shapes[0]) != 3 or len(shapes[1]) != 3):
            raise RuntimeError(f"Piper {self.lang}: frontend remainder must declare [z, y_mask]")
        self._frontend_manifest = manifest
        try:
            self._frontend_rknn = RKNNLite(verbose=False)
            if self._frontend_rknn.load_rknn(str(rknn_path)) != 0 or self._frontend_rknn.init_runtime() != 0:
                raise RuntimeError(f"Failed to initialize Piper frontend RKNN for {self.lang}")
            self._rknn = RKNNLite(verbose=False)
            decoder_path = self.model_dir / "flow_decoder.rknn"
            if self._rknn.load_rknn(str(decoder_path)) != 0 or self._rknn.init_runtime() != 0:
                raise RuntimeError(f"Failed to initialize Piper decoder RKNN for {self.lang}")
            self._remainder = ort.InferenceSession(str(remainder_path), providers=["CPUExecutionProvider"])
            self._frontend_npu = True
            self.mel_len = self._probe_decoder_window()
        except Exception:
            self.release()
            raise
        self._hybrid = False
        logger.info("Loaded Piper frontend NPU for %s (bucket=%d; remainder=ORT CPU)", self.lang, self.seq_len)

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
        candidates = ([int(env)] if env.isdigit() and int(env) > 0 else []) + [
            self.mel_len, 512, 256, 150, 128, 1024]
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
        if self._frontend_rknn is not None:
            try:
                self._frontend_rknn.release()
            except Exception:
                pass
            self._frontend_rknn = None
        self._encoder = None
        self._remainder = None
        self._frontend_npu = False

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
        if self._frontend_npu:
            return self._infer_frontend_npu(token_ids, length_scale, noise_scale, noise_w)
        return self._infer_legacy(token_ids, length_scale, noise_scale, noise_w)

    def _infer_frontend_npu(self, token_ids: list[int], length_scale: float,
                            noise_scale: float, noise_w: float) -> np.ndarray:
        if len(token_ids) > self.seq_len:
            raise ValueError(
                f"Piper {self.lang}: frontend NPU supports at most {self.seq_len} "
                f"tokens, got {len(token_ids)}; split the text or use a larger bucket"
            )
        n = min(len(token_ids), self.seq_len)
        tokens = np.zeros((1, self.seq_len), dtype=np.int64); tokens[0, :n] = token_ids[:n]
        lengths = np.array([n], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
        x_mask = np.zeros((1, 1, self.seq_len), dtype=np.float32); x_mask[0, 0, :n] = 1.0
        input_values = {"input": tokens, "input_lengths": lengths,
                        "scales": scales, "x_mask": x_mask,
                        "sid": np.array([0], dtype=np.int64)}
        rknn_inputs = self._frontend_manifest["prefix"].get("inputs", [])
        if not rknn_inputs:
            raise RuntimeError(f"Piper {self.lang}: frontend manifest has no inputs")
        rknn_names = [x["name"] if isinstance(x, dict) else x for x in rknn_inputs]
        try:
            out = self._frontend_rknn.inference(inputs=[input_values[name] for name in rknn_names])
        except KeyError as exc:
            raise RuntimeError(f"Piper {self.lang}: unsupported frontend input {exc}") from exc
        names = [x["name"] for x in self._frontend_manifest["prefix"]["outputs"]]
        if out is None or len(out) != len(names):
            raise RuntimeError(f"Piper {self.lang}: frontend RKNN output count mismatch")
        feeds = {"input": tokens, "input_lengths": lengths, "scales": scales, "x_mask": x_mask}
        feeds.update(dict(zip(names, out)))
        dtype_map = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                     "tensor(float)": np.float32, "tensor(float16)": np.float16}
        def declared_shape(inp, name):
            dims = []
            for dim in inp.shape:
                if isinstance(dim, int) and dim > 0:
                    dims.append(dim)
                elif isinstance(dim, str) and any(k in dim.lower() for k in ("seq", "phoneme", "token")):
                    dims.append(self.seq_len)
                else:
                    raise RuntimeError(f"Piper {self.lang}: unsupported symbolic shape for {name}: {inp.shape}")
            return tuple(dims)
        for inp in self._remainder.get_inputs():
            if inp.name == "audio_length":
                feeds[inp.name] = np.zeros(declared_shape(inp, inp.name), dtype=dtype_map.get(inp.type, np.float32))
            elif inp.name == "cumulative_durations":
                feeds[inp.name] = np.zeros(declared_shape(inp, inp.name), dtype=dtype_map.get(inp.type, np.float32))
        declared = {x.name for x in self._remainder.get_inputs()}
        feeds = {name: value for name, value in feeds.items() if name in declared}
        for inp in self._remainder.get_inputs():
            if inp.name == "sid": feeds[inp.name] = np.array([0], dtype=np.int64)
        result = self._remainder.run(None, feeds)
        if len(result) != 2:
            raise RuntimeError(f"Piper {self.lang}: frontend remainder returned no audio")
        z, y_mask = result
        if (z.ndim != 3 or y_mask.ndim != 3 or z.shape[0] != y_mask.shape[0] or
                z.shape[2] != y_mask.shape[2] or z.shape[1] != 192 or y_mask.shape[1] != 1 or
                not np.isfinite(z).all() or not np.isfinite(y_mask).all()):
            raise RuntimeError(f"Piper {self.lang}: remainder outputs are not z/y_mask shapes")
        return self._decode_mel(np.asarray(z), np.asarray(y_mask), int(z.shape[2]))

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
        n = min(len(token_ids), self.seq_len)
        tokens = np.zeros((1, self.seq_len), dtype=np.int64)
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

# Same boundary rule as _CLAUSE_END_RE, for the marks that end a segment. The
# old pattern split after ANY "." -- inside "3.14", "example.com" and after
# "Dr." -- and each of those stray segments now also earns a sentence pause, so
# the two splitters have to agree on what a boundary is. Closing quotes and
# brackets stay with the sentence they close.
_SENTENCE_END_RE = re.compile(
    r'([.!?;]+)(?=[\s"\'”’)\]]|$)["\'”’)\]]*|([。！？；]+)[”’）」』]*|\n+'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming synthesis."""
    text = text.strip()
    parts: list[str] = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(text):
        if m.group(1) is not None and _is_abbreviation_dot(text, m):
            continue
        parts.append(text[pos:m.end()])
        pos = m.end()
    parts.append(text[pos:])
    return [p.strip() for p in parts if p.strip()]


def _frontend_tokens(text: str, lang_model, cache: dict | None = None) -> tuple[str, list[int]]:
    key = (id(lang_model), text)
    if cache is not None and key in cache:
        return cache[key]
    espeak_cache = None
    if cache is not None:
        espeak_cache = cache.setdefault(("__piper_espeak_cache__", id(lang_model)), {})
    phonemes = text_to_phonemes(
        text, lang_model.espeak_voice, espeak_cache=espeak_cache
    )
    value = (phonemes, phonemes_to_ids(phonemes, lang_model.phoneme_id_map))
    if cache is not None:
        cache[key] = value
    return value


def _frontend_token_count(text: str, lang_model, cache: dict | None = None) -> int:
    return len(_frontend_tokens(text, lang_model, cache)[1])


_FRONTEND_NATURAL_PROBE_LIMIT = 8
_FRONTEND_HARD_PROBE_LIMIT = 8
_FRONTEND_WINDOW_CHARS_PER_TOKEN = 8
_FRONTEND_MIN_WINDOW_CHARS = 32


def _split_frontend_bucket(text: str, lang_model, cache: dict | None = None):
    """Split one sentence by measured token count for a fixed frontend bucket."""
    cap = getattr(lang_model, "seq_len", SEQ_LEN)
    current = text
    while current.strip():
        window_limit = max(
            _FRONTEND_MIN_WINDOW_CHARS,
            cap * _FRONTEND_WINDOW_CHARS_PER_TOKEN,
        )
        window_end = min(len(current), window_limit)
        window_text = current[:window_end]
        # Never phonemize the complete remaining tail when it exceeds the
        # window. Every chosen prefix is still measured with the real tokenizer.
        window_count = _frontend_token_count(window_text, lang_model, cache)
        if window_end == len(current) and window_count <= cap:
            yield current
            return
        if len(current) <= 1:
            raise ValueError(
                f"Piper: one minimal text unit has {window_count} tokens, exceeds frontend "
                f"bucket {cap}"
            )

        candidates = [
            index + 1 for index, char in enumerate(window_text[:-1])
            if char in ",，;；" and index + 1 < len(current)
        ]
        candidates.extend(
            index + 1 for index, char in enumerate(window_text[:-1])
            if char.isspace() and index + 1 < len(current)
        )
        split_at = None
        # Estimate only orders the measured candidates; it is not a monotonic
        # token-count assumption.
        estimate = max(
            1,
            min(
                window_end - 1,
                int(window_end * cap / max(window_count, 1)),
            ),
        )
        natural = sorted(
            set(candidates), key=lambda index: (abs(index - estimate), -index)
        )[:_FRONTEND_NATURAL_PROBE_LIMIT]
        for index in natural:
            if current[:index].strip() and _frontend_token_count(current[:index], lang_model, cache) <= cap:
                split_at = index
                break
        if split_at is None and window_count <= cap:
            # The whole measured window fits, but no natural boundary did.
            # Keep the exact measured window rather than cutting the tail.
            split_at = window_end
        if split_at is None:
            # Last resort: Unicode character boundary.  Sample around a
            # density-based estimate; token count is still measured for every
            # candidate, without assuming it is monotonic.
            stride = max(1, window_end // _FRONTEND_HARD_PROBE_LIMIT)
            probes = []
            index = estimate
            while index > 0 and len(probes) < _FRONTEND_HARD_PROBE_LIMIT:
                probes.append(index)
                index -= stride
            # Keep leading whitespace with the first non-whitespace unit so the
            # fallback always makes progress without emitting a blank segment.
            first_nonblank = len(current) - len(current.lstrip()) + 1
            if first_nonblank <= window_end:
                probes.append(first_nonblank)
            for index in probes:
                if not current[:index].strip() or index >= len(current):
                    continue
                if _frontend_token_count(current[:index], lang_model, cache) <= cap:
                    split_at = index
                    break
        if split_at is None:
            raise ValueError(
                f"Piper: cannot fit a minimal text unit into frontend bucket {cap} "
                f"(measured {window_count} tokens)"
            )
        prefix, current = current[:split_at], current[split_at:]
        if prefix.strip():
            yield prefix


def _frontend_segments(text: str, lang_model, cache: dict | None = None):
    # Every Piper _LangModel has a fixed phoneme input shape, whether its
    # encoder is CPU hybrid, legacy RKNN, or the opt-in frontend split. Keep
    # Kokoro's own dynamic CPU path on ordinary sentence splitting.
    if not (
        isinstance(lang_model, _LangModel)
        or getattr(lang_model, "_frontend_npu", False)
        or hasattr(lang_model, "seq_len")
    ):
        yield from _split_sentences(text)
        return
    for sentence in _split_sentences(text):
        yield from _split_frontend_bucket(sentence, lang_model, cache)


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
        self._chinese_fallback = None
        fallback_setting = os.environ.get("PIPER_CHINESE_FALLBACK", "").strip().lower()
        if fallback_setting not in {"", "off", "0", "false", "none", "disabled", "matcha_rknn"}:
            raise ValueError(
                "PIPER_CHINESE_FALLBACK must be empty/off or 'matcha_rknn', "
                f"got {fallback_setting!r}"
            )
        self._chinese_fallback_enabled = fallback_setting == "matcha_rknn"

    # ------------------------------------------------------------------
    # TTSBackend protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "piper_rknn"

    def is_ready(self) -> bool:
        if not (self._ready and bool(self._models)):
            return False
        if getattr(self, "_chinese_fallback_enabled", False):
            fallback = self._chinese_fallback
            return fallback is not None and bool(
                getattr(fallback, "is_ready", lambda: True)()
            )
        return True

    def runtime_info(self) -> dict:
        """Return backend readiness and rates without exposing model paths."""
        english_ready = bool(self._ready and self._models)
        chinese_ready = False
        if self._chinese_fallback_enabled and self._chinese_fallback is not None:
            chinese_ready = bool(
                getattr(self._chinese_fallback, "is_ready", lambda: True)()
            )
        return {
            "backend": self.name,
            "english_ready": english_ready,
            "chinese_fallback_enabled": self._chinese_fallback_enabled,
            "chinese_ready": chinese_ready,
            "sample_rate": self.get_sample_rate(),
        }

    def get_sample_rate(self) -> int:
        # Return sample rate of default lang if loaded, else 22050
        if DEFAULT_LANG in self._models:
            return self._models[DEFAULT_LANG].sample_rate
        if self._models:
            return next(iter(self._models.values())).sample_rate
        return SAMPLE_RATE

    @staticmethod
    def _require_soxr():
        try:
            import soxr
        except ImportError as exc:
            raise RuntimeError(
                "PIPER_CHINESE_FALLBACK=matcha_rknn requires the "
                "piper-bilingual optional dependency (soxr)"
            ) from exc
        return soxr

    def _load_chinese_fallback(self) -> None:
        self._require_soxr()
        from rkvoice_stream.engine.tts import create_backend

        fallback = create_backend("matcha_rknn")
        # Keep the partially-created backend reachable until preload succeeds,
        # so a failed Matcha load is rolled back by the same cleanup path as
        # the Piper models.
        self._chinese_fallback = fallback
        try:
            fallback.preload()
        except Exception:
            self._chinese_fallback = None
            cleanup = getattr(fallback, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    logger.exception("Failed to roll back Chinese fallback")
            raise

    @staticmethod
    def _language_route(text: str, language: Optional[str]) -> str:
        value = (language or "").strip().lower()
        if value in {"", "auto", "detect", "default"}:
            # Mixed CJK/Latin text stays on the Chinese path when the caller
            # leaves language selection to auto detection.  The generic
            # detector is ratio based and would otherwise classify short
            # mixed prompts as English.
            if any(
                0x3400 <= ord(ch) <= 0x4DBF
                or 0x4E00 <= ord(ch) <= 0x9FFF
                or 0x20000 <= ord(ch) <= 0x2A6DF
                or 0xF900 <= ord(ch) <= 0xFAFF
                for ch in text
            ):
                return "zh"
            detected = detect_language(text)
            if detected.startswith("zh"):
                return "zh"
            if detected.startswith("en"):
                return "en"
            raise ValueError(f"Piper Chinese fallback cannot route detected language {detected!r}")
        if value in {"zh", "zh-cn", "zh_cn", "zh-hans", "zh_hans", "chinese", "mandarin", "cn"}:
            return "zh"
        if value in {"en", "en-us", "en_us", "en-gb", "english", "us"}:
            return "en"
        raise ValueError(f"Piper Chinese fallback does not support language {language!r}")

    def _matcha_audio_to_piper_wav(self, wav_bytes: bytes) -> tuple[bytes, dict]:
        import soundfile as sf

        source_audio, source_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if source_audio.ndim > 1:
            source_audio = source_audio[:, 0]
        source_audio = np.asarray(source_audio, dtype=np.float32)
        target_rate = self.get_sample_rate()
        if int(source_rate) != target_rate:
            soxr = self._require_soxr()
            source_audio = soxr.resample(
                source_audio, int(source_rate), target_rate, quality="HQ"
            ).astype(np.float32, copy=False)
        out = io.BytesIO()
        sf.write(out, source_audio, target_rate, format="WAV", subtype="PCM_16")
        return out.getvalue(), {
            "backend": "matcha_rknn",
            "language": "zh",
            "sample_rate": target_rate,
            "source_sample_rate": int(source_rate),
            "duration": len(source_audio) / target_rate,
        }

    def _synthesize_chinese(
        self, text: str, speed: Optional[float], pitch_shift: Optional[float], kwargs: dict
    ) -> tuple[bytes, dict]:
        if self._chinese_fallback is None:
            raise RuntimeError("Chinese fallback is not loaded")
        synth_kwargs = {
            k: v for k, v in kwargs.items() if k == "noise_scale" and v is not None
        }
        started = time.perf_counter()
        wav_bytes, meta = self._chinese_fallback.synthesize(
            text=text,
            speaker_id=0,
            speed=speed,
            pitch_shift=pitch_shift,
            **synth_kwargs,
        )
        resample_started = time.perf_counter()
        wav_bytes, route_meta = self._matcha_audio_to_piper_wav(wav_bytes)
        elapsed = time.perf_counter() - started
        route_meta["resample_ms"] = (time.perf_counter() - resample_started) * 1000
        route_meta["inference_time"] = elapsed
        route_meta["rtf"] = (
            elapsed / route_meta["duration"] if route_meta["duration"] > 0 else 0.0
        )
        return wav_bytes, {**meta, **route_meta}

    def _synthesize_chinese_stream(
        self, text: str, speed: Optional[float], pitch_shift: Optional[float], kwargs: dict
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if self._chinese_fallback is None:
            raise RuntimeError("Chinese fallback is not loaded")
        soxr = self._require_soxr()
        source_rate = int(self._chinese_fallback.get_sample_rate())
        target_rate = self.get_sample_rate()
        resampler = soxr.ResampleStream(
            source_rate, target_rate, 1, dtype="float32", quality="HQ"
        )
        synth_kwargs = {
            k: v for k, v in kwargs.items() if k == "noise_scale" and v is not None
        }
        source = self._chinese_fallback.synthesize_stream(
            text=text,
            speaker_id=0,
            speed=speed,
            pitch_shift=pitch_shift,
            **synth_kwargs,
        )
        completed = False
        try:
            for audio, meta in source:
                converted = resampler.resample_chunk(
                    np.asarray(audio, dtype=np.float32), last=False
                )
                if len(converted) == 0:
                    continue
                out_meta = {
                    **meta,
                    "backend": "matcha_rknn",
                    "language": "zh",
                    "sample_rate": target_rate,
                    "source_sample_rate": source_rate,
                    "duration": len(converted) / target_rate,
                }
                yield converted.astype(np.float32, copy=False), out_meta
            tail = resampler.resample_chunk(
                np.empty(0, dtype=np.float32), last=True
            )
            if len(tail) > 0:
                yield tail.astype(np.float32, copy=False), {
                    "backend": "matcha_rknn",
                    "language": "zh",
                    "sample_rate": target_rate,
                    "source_sample_rate": source_rate,
                    "duration": len(tail) / target_rate,
                }
            completed = True
        finally:
            if not completed and hasattr(source, "close"):
                source.close()

    def preload(self) -> None:
        """Load RKNN models for all configured languages.

        Japanese (ja/ja_JP) is handled by sherpa-onnx Kokoro v1.0 on CPU
        instead of RKNN NPU (vocoder dimension exceeds NPU limit).
        """
        try:
            model_root = Path(MODEL_DIR)
            for lang in PRELOAD_LANGS:
                lang_dir = model_root / lang
                if not lang_dir.exists():
                    logger.warning("Piper model dir not found for %s: %s", lang, lang_dir)
                    continue
                try:
                    if lang in _JA_LANGS or lang.split("_")[0] == "ja":
                        m = _JaKokoroModel(lang, lang_dir)
                        m.load()
                    else:
                        m = _LangModel(lang, lang_dir)
                        m.load()
                    self._models[lang] = m
                except Exception as exc:
                    logger.error("Failed to load model for %s: %s", lang, exc)

            if not self._models:
                logger.error("PiperRKNNBackend: no models loaded — backend not ready")
                return
            if self._chinese_fallback_enabled:
                self._load_chinese_fallback()
                if not bool(getattr(self._chinese_fallback, "is_ready", lambda: True)()):
                    raise RuntimeError(
                        "Chinese fallback preload completed but backend is not ready"
                    )
            self._ready = True
            logger.info(
                "PiperRKNNBackend ready. Loaded languages: %s%s",
                list(self._models.keys()),
                "; Chinese fallback=matcha_rknn" if self._chinese_fallback else "",
            )
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> None:
        """Release all RKNN contexts."""
        for m in self._models.values():
            m.release()
        self._models.clear()
        if self._chinese_fallback is not None:
            cleanup = getattr(self._chinese_fallback, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    logger.exception("Failed to clean up Chinese fallback")
            self._chinese_fallback = None
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
        token_cache: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Synthesize a single text segment. Returns (audio_float32, meta).

        Dispatches to CPU Kokoro path for Japanese, RKNN path for all others.
        """
        if isinstance(lang_model, _JaKokoroModel):
            return self._synthesize_segment_ja(text, lang_model, speed)

        meta: dict = {}

        t0 = time.perf_counter()
        try:
            phoneme_str, cached_ids = _frontend_tokens(text, lang_model, token_cache)
        except RuntimeError as exc:
            logger.error("Phonemization failed: %s", exc)
            return np.zeros(0, dtype=np.float32), {"error": str(exc)}
        meta["phonemize_ms"] = (time.perf_counter() - t0) * 1000

        token_ids = cached_ids
        meta["num_tokens"] = len(token_ids)

        if not token_ids:
            logger.warning("No token IDs for text %r (phonemes: %r)", text, phoneme_str)
            return np.zeros(0, dtype=np.float32), meta

        # The caller normally bounded the segment using the actual model
        # shape. Direct calls fail closed rather than silently dropping text.
        seq_len = getattr(lang_model, "seq_len", SEQ_LEN)
        if len(token_ids) > seq_len:
            raise ValueError(
                f"Piper: text segment exceeds model seq_len={seq_len}; "
                f"frontend NPU supports at most {seq_len} tokens, got {len(token_ids)}; "
                "split the text or use a larger bucket"
            )

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

        if getattr(self, "_chinese_fallback_enabled", False):
            route = self._language_route(text, language)
            if route == "zh":
                return self._synthesize_chinese(text, speed, pitch_shift, kwargs)

        # Resolve language
        effective_lang = language or detect_language(text)
        lang_model = self._get_model(effective_lang)

        effective_speed = speed if speed is not None else 1.0
        noise_scale = kwargs.get("noise_scale", None)

        t_start = time.perf_counter()

        # Split into sentences for long texts
        token_cache = {}
        sentences = _frontend_segments(text, lang_model, token_cache)
        all_audio: list[np.ndarray] = []
        agg_meta: dict = {
            "phonemize_ms": 0.0,
            "segmentation_ms": 0.0,
            "infer_ms": 0.0,
            "total_ms": 0.0,
            "num_tokens": 0,
        }

        while True:
            segment_start = time.perf_counter()
            try:
                sentence = next(sentences)
            except StopIteration:
                break
            agg_meta["segmentation_ms"] += (time.perf_counter() - segment_start) * 1000
            audio_seg, seg_meta = self._synthesize_segment(
                sentence, lang_model, effective_speed, noise_scale,
                token_cache=token_cache,
            )
            if len(audio_seg) > 0:
                all_audio.append(audio_seg)
            for k in ("phonemize_ms", "infer_ms", "total_ms", "num_tokens"):
                agg_meta[k] = agg_meta.get(k, 0.0) + seg_meta.get(k, 0.0)
        agg_meta["total_ms"] += agg_meta["segmentation_ms"]

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
            "sample_rate": lang_model.sample_rate,
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

        if getattr(self, "_chinese_fallback_enabled", False):
            route = self._language_route(text, language)
            if route == "zh":
                yield from self._synthesize_chinese_stream(
                    text, speed, pitch_shift, kwargs
                )
                return

        effective_lang = language or detect_language(text)
        lang_model = self._get_model(effective_lang)
        effective_speed = speed if speed is not None else 1.0
        noise_scale = kwargs.get("noise_scale", None)

        token_cache = {}
        sentences = _frontend_segments(text, lang_model, token_cache)
        while True:
            segment_start = time.perf_counter()
            try:
                sentence = next(sentences)
            except StopIteration:
                break
            segmentation_ms = (time.perf_counter() - segment_start) * 1000
            audio_seg, seg_meta = self._synthesize_segment(
                sentence, lang_model, effective_speed, noise_scale,
                token_cache=token_cache,
            )
            if len(audio_seg) == 0:
                continue

            peak = np.abs(audio_seg).max()
            if peak > 0:
                audio_seg = audio_seg / peak * 0.95

            duration = len(audio_seg) / lang_model.sample_rate
            meta = {
                "duration": duration,
                "sample_rate": lang_model.sample_rate,
                "language": effective_lang,
                "segmentation_ms": segmentation_ms,
                **seg_meta,
            }
            meta["total_ms"] = seg_meta.get("total_ms", 0.0) + segmentation_ms
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
