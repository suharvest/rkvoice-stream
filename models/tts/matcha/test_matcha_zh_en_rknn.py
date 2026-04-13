#!/usr/bin/env python3
"""
Matcha-Icefall-ZH-EN TTS Inference for RK3576 (RKNN + CPU)

This script provides a complete TTS pipeline for matcha-icefall-zh-en on RK3576:
1. Text frontend: Convert Chinese/English text to phonemes using espeak-ng
2. Acoustic model (RKNN): Generate mel spectrogram from phonemes
3. Vocoder (RKNN): Convert mel to STFT components
4. ISTFT (CPU): Reconstruct waveform from STFT

Usage on RK3576:
    # Test with default text
    python test_matcha_zh_en_rknn.py

    # Test with custom text
    python test_matcha_zh_en_rknn.py --text "你好世界"

    # Enable verbose output
    python test_matcha_zh_en_rknn.py --verbose

Requirements:
    - rknnlite (runtime 2.3.0)
    - piper-phonemizer (for text frontend)
    - numpy, soundfile

Files needed:
    - matcha-zh-en.rknn or matcha-zh-en-int8.rknn
    - vocos-16khz-univ.rknn or vocos-16khz-univ-int8.rknn
    - Text frontend files: lexicon.txt, tokens.txt, espeak-ng-data/, *.fst
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Try to import RKNN
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("Warning: rknnlite not found, using ONNX Runtime fallback")

# Try to import ONNX Runtime for fallback
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Try to import soundfile
try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

# ============================================================================
# Configuration
# ============================================================================

# Default paths (adjust for your setup)
DEFAULT_MODEL_DIR = Path("/home/cat/matcha-icefall-zh-en-rknn")
DEFAULT_ACOUSTIC_RKNN = DEFAULT_MODEL_DIR / "matcha-zh-en.rknn"
DEFAULT_VOCODER_RKNN = DEFAULT_MODEL_DIR / "vocos-16khz-univ-int8.rknn"
DEFAULT_FRONTEND_DIR = DEFAULT_MODEL_DIR / "matcha-icefall-zh-en"

# Audio parameters
SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_FFT = 1024
NUM_MEL_BANDS = 80

# Model parameters
MAX_SEQ_LEN = 96
NOISE_SCALE = 0.667
DEFAULT_SPEED = 1.0


# ============================================================================
# Text Frontend
# ============================================================================

class TextFrontend:
    """Convert text to phoneme IDs for matcha-icefall-zh-en."""

    def __init__(self, frontend_dir: Path):
        self.frontend_dir = Path(frontend_dir)
        self.tokens = {}
        self.lexicon = {}

        # Load tokens
        tokens_path = self.frontend_dir / "tokens.txt"
        if tokens_path.exists():
            with open(tokens_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.tokens[parts[0]] = int(parts[1])

        # Load lexicon (word -> phonemes)
        lexicon_path = self.frontend_dir / "lexicon.txt"
        if lexicon_path.exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.lexicon[parts[0]] = parts[1:]

        print(f"Loaded {len(self.tokens)} tokens, {len(self.lexicon)} lexicon entries")

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.

        For matcha-icefall-zh-en, this uses espeak-ng for English and
        a rule-based approach for Chinese.
        """
        phonemes = []

        # Simple approach: use lexicon for known words
        # For production, use piper-phonemize or espeak-ng directly

        for char in text:
            if char in self.lexicon:
                phonemes.extend(self.lexicon[char])
            elif char in self.tokens:
                phonemes.append(char)
            else:
                # Unknown character - try to split into pinyin
                # This is a simplified approach; production should use proper pinyin conversion
                phonemes.append(char)

        return phonemes

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to token IDs."""
        ids = []
        for p in phonemes:
            if p in self.tokens:
                ids.append(self.tokens[p])
            else:
                # Unknown token - use UNK (usually 0 or last token)
                ids.append(0)
        return ids


def text_to_token_ids_simple(text: str, frontend_dir: Path) -> np.ndarray:
    """
    Simple text-to-IDs conversion.

    For production use, integrate piper-phonemize:
        pip install piper-phonemize

    Or use sherpa-onnx's built-in frontend:
        import sherpa_onnx
        # See sherpa-onnx documentation for text frontend
    """
    frontend = TextFrontend(frontend_dir)
    phonemes = frontend.text_to_phonemes(text)
    ids = frontend.phonemes_to_ids(phonemes)

    # Pad to MAX_SEQ_LEN
    tokens = np.zeros((1, MAX_SEQ_LEN), dtype=np.int64)
    tokens[0, :min(len(ids), MAX_SEQ_LEN)] = ids[:MAX_SEQ_LEN]

    return tokens, min(len(ids), MAX_SEQ_LEN)


# ============================================================================
# ISTFT (Inverse Short-Time Fourier Transform)
# ============================================================================

def istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray,
          n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Convert STFT magnitude and phase components to waveform.

    Args:
        mag: Magnitude [n_freqs, n_frames]
        x: Cos component (real) [n_freqs, n_frames]
        y: Sin component (imag) [n_freqs, n_frames]
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        waveform: Audio samples
    """
    # Reconstruct complex spectrogram
    complex_spec = mag * (x + 1j * y)

    n_frames = complex_spec.shape[-1]
    output_len = (n_frames - 1) * hop_length + n_fft
    waveform = np.zeros(output_len, dtype=np.float32)
    window = np.hanning(n_fft)

    # Overlap-add ISTFT
    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[..., i], n=n_fft) * window
        start = i * hop_length
        waveform[start:start + n_fft] += frame

    # Normalize by window sum
    window_sum = np.zeros(output_len, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        window_sum[start:start + n_fft] += window ** 2

    waveform = waveform / np.maximum(window_sum, 1e-8)

    return waveform


# ============================================================================
# TTS Engine
# ============================================================================

class MatchaTTSEngine:
    """Matcha-icefall-zh-en TTS engine using RKNN."""

    def __init__(self, acoustic_path: Path, vocoder_path: Path, frontend_dir: Path):
        self.acoustic_path = Path(acoustic_path)
        self.vocoder_path = Path(vocoder_path)
        self.frontend_dir = Path(frontend_dir)

        self.acoustic_model = None
        self.vocoder_model = None
        self.frontend = None

    def load(self):
        """Load all models."""
        print("Loading models...")

        # Load text frontend
        self.frontend = TextFrontend(self.frontend_dir)

        if HAS_RKNN:
            # Load acoustic RKNN
            print(f"  Loading acoustic model: {self.acoustic_path}")
            self.acoustic_model = RKNNLite(verbose=False)
            ret = self.acoustic_model.load_rknn(str(self.acoustic_path))
            if ret != 0:
                raise RuntimeError(f"Failed to load acoustic RKNN: ret={ret}")
            ret = self.acoustic_model.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"Failed to init acoustic runtime: ret={ret}")
            print("    ✓ Acoustic model loaded")

            # Load vocoder RKNN
            print(f"  Loading vocoder: {self.vocoder_path}")
            self.vocoder_model = RKNNLite(verbose=False)
            ret = self.vocoder_model.load_rknn(str(self.vocoder_path))
            if ret != 0:
                raise RuntimeError(f"Failed to load vocoder RKNN: ret={ret}")
            ret = self.vocoder_model.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"Failed to init vocoder runtime: ret={ret}")
            print("    ✓ Vocoder loaded")

        elif HAS_ORT:
            # Fallback to ONNX Runtime
            print("  Using ONNX Runtime fallback")
            acoustic_onnx = self.acoustic_path.with_suffix('.onnx')
            vocoder_onnx = self.vocoder_path.with_suffix('.onnx')

            if acoustic_onnx.exists():
                self.acoustic_model = ort.InferenceSession(str(acoustic_onnx))
            if vocoder_onnx.exists():
                self.vocoder_model = ort.InferenceSession(str(vocoder_onnx))

        else:
            raise RuntimeError("Neither rknnlite nor onnxruntime available")

    def synthesize(self, text: str, speed: float = DEFAULT_SPEED) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.

        Args:
            text: Input text (Chinese or English)
            speed: Speech speed (1.0 = normal)

        Returns:
            audio: Audio samples (float32)
            metadata: Timing and other info
        """
        metadata = {}

        # Step 1: Text to phonemes
        t0 = time.perf_counter()
        tokens, x_length = text_to_token_ids_simple(text, self.frontend_dir)
        metadata["text_frontend_ms"] = (time.perf_counter() - t0) * 1000
        metadata["num_tokens"] = x_length

        if x_length == 0:
            return np.zeros(0, dtype=np.float32), metadata

        # Step 2: Acoustic model (tokens -> mel)
        t0 = time.perf_counter()
        noise_scale = np.array([NOISE_SCALE], dtype=np.float32)
        length_scale = np.array([speed], dtype=np.float32)
        x_len = np.array([x_length], dtype=np.int64)

        if HAS_RKNN:
            mel = self.acoustic_model.inference(
                inputs=[tokens, x_len, noise_scale, length_scale]
            )[0]
        else:
            mel = self.acoustic_model.run(
                None,
                {"tokens": tokens, "x_length": x_len,
                 "noise_scale": noise_scale, "length_scale": length_scale}
            )[0]

        metadata["acoustic_ms"] = (time.perf_counter() - t0) * 1000
        metadata["mel_shape"] = list(mel.shape)

        # Trim mel to expected length
        mel_frames = int(x_length * 30 * speed + 0.5)
        mel_frames = min(mel_frames, mel.shape[2])
        mel_trimmed = mel[:, :, :mel_frames]

        # Step 3: Pad mel for vocoder
        vocoder_frames = 256  # Fixed for RKNN
        mel_padded = np.zeros((1, NUM_MEL_BANDS, vocoder_frames), dtype=np.float32)
        mel_padded[:, :, :min(mel_frames, vocoder_frames)] = mel_trimmed[:, :, :min(mel_frames, vocoder_frames)]

        # Step 4: Vocoder (mel -> STFT)
        t0 = time.perf_counter()
        if HAS_RKNN:
            outputs = self.vocoder_model.inference(inputs=[mel_padded])
        else:
            outputs = self.vocoder_model.run(None, {"mels": mel_padded})

        metadata["vocoder_ms"] = (time.perf_counter() - t0) * 1000

        # Vocos outputs: mag, x (cos), y (sin)
        mag = outputs[0][0]  # [513, T]
        x = outputs[1][0]    # [513, T]
        y = outputs[2][0]    # [513, T]

        # Step 5: ISTFT (STFT -> waveform)
        t0 = time.perf_counter()
        audio = istft(mag, x, y, N_FFT, HOP_LENGTH)
        metadata["istft_ms"] = (time.perf_counter() - t0) * 1000

        # Trim to expected length
        expected_samples = mel_frames * HOP_LENGTH
        audio = audio[:expected_samples]

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        metadata["duration_s"] = len(audio) / SAMPLE_RATE
        metadata["total_ms"] = sum(v for k, v in metadata.items() if k.endswith("_ms"))

        if metadata["duration_s"] > 0:
            metadata["rtf"] = metadata["total_ms"] / 1000 / metadata["duration_s"]
        else:
            metadata["rtf"] = 0

        return audio.astype(np.float32), metadata

    def cleanup(self):
        """Release resources."""
        if self.acoustic_model:
            try:
                self.acoustic_model.release()
            except:
                pass
        if self.vocoder_model:
            try:
                self.vocoder_model.release()
            except:
                pass


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Matcha-Icefall-ZH-EN TTS for RK3576")
    parser.add_argument("--text", "-t", default="你好世界", help="Text to synthesize")
    parser.add_argument("--output", "-o", default="/tmp/matcha_output.wav", help="Output WAV file")
    parser.add_argument("--acoustic", default=str(DEFAULT_ACOUSTIC_RKNN), help="Acoustic RKNN path")
    parser.add_argument("--vocoder", default=str(DEFAULT_VOCODER_RKNN), help="Vocoder RKNN path")
    parser.add_argument("--frontend", default=str(DEFAULT_FRONTEND_DIR), help="Frontend directory")
    parser.add_argument("--speed", "-s", type=float, default=DEFAULT_SPEED, help="Speech speed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-phonemes", type=int, default=96, help="Max phoneme sequence length (bucket size)")

    args = parser.parse_args()

    global MAX_SEQ_LEN
    MAX_SEQ_LEN = args.max_phonemes

    # Check files
    acoustic_path = Path(args.acoustic)
    vocoder_path = Path(args.vocoder)
    frontend_dir = Path(args.frontend)

    if not acoustic_path.exists():
        # Try ONNX fallback
        acoustic_onnx = acoustic_path.with_suffix('.onnx')
        if acoustic_onnx.exists():
            acoustic_path = acoustic_onnx
            print(f"Using ONNX fallback: {acoustic_path}")
        else:
            print(f"ERROR: Acoustic model not found: {acoustic_path}")
            return 1

    if not vocoder_path.exists():
        vocoder_onnx = vocoder_path.with_suffix('.onnx')
        if vocoder_onnx.exists():
            vocoder_path = vocoder_onnx
            print(f"Using ONNX fallback: {vocoder_path}")
        else:
            print(f"ERROR: Vocoder not found: {vocoder_path}")
            return 1

    # Create engine
    engine = MatchaTTSEngine(acoustic_path, vocoder_path, frontend_dir)

    try:
        engine.load()

        print(f"\nSynthesizing: {args.text}")

        audio, metadata = engine.synthesize(args.text, args.speed)

        print(f"\nResults:")
        print(f"  Tokens: {metadata.get('num_tokens', 0)}")
        print(f"  Duration: {metadata.get('duration_s', 0):.2f}s")
        print(f"  Text frontend: {metadata.get('text_frontend_ms', 0):.1f}ms")
        print(f"  Acoustic model: {metadata.get('acoustic_ms', 0):.1f}ms")
        print(f"  Vocoder: {metadata.get('vocoder_ms', 0):.1f}ms")
        print(f"  ISTFT: {metadata.get('istft_ms', 0):.1f}ms")
        print(f"  Total: {metadata.get('total_ms', 0):.1f}ms")
        print(f"  RTF: {metadata.get('rtf', 0):.3f}")

        # Save audio
        if HAS_SF:
            sf.write(args.output, audio, SAMPLE_RATE)
            print(f"\nSaved to: {args.output}")
        else:
            print(f"\nNote: soundfile not installed, skipping save")

        if args.verbose:
            print(f"\nMetadata: {json.dumps(metadata, indent=2)}")

    finally:
        engine.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())