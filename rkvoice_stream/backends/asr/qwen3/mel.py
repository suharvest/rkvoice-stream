"""Mel spectrogram feature extractor (pure NumPy, no librosa dependency)."""

import numpy as np


def _stft_numpy(audio: np.ndarray, n_fft: int = 400,
                hop_length: int = 160, center: bool = True) -> np.ndarray:
    """
    Short-time Fourier Transform using pure NumPy.

    Equivalent to ``librosa.stft(audio, n_fft=400, hop_length=160,
    window='hann', center=True)`` but without the librosa dependency.

    Returns:
        Complex STFT matrix of shape (n_fft//2 + 1, n_frames).
    """
    if center:
        pad_len = n_fft // 2
        audio = np.pad(audio, (pad_len, pad_len), mode='reflect')

    window = np.hanning(n_fft + 1)[:n_fft].astype(audio.dtype)

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    # Build frame indices and extract all frames at once
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = audio[idx] * window  # (n_frames, n_fft)
    return np.fft.rfft(frames, n=n_fft, axis=1).T  # (n_fft//2+1, n_frames)


class MelExtractor:
    """
    Whisper-compatible mel spectrogram extractor.

    Uses pre-computed mel filter bank (from Qwen3-ASR / Whisper).
    Parameters: n_fft=400, hop_length=160, window='hann', center=True.
    Output: (128, T) mel spectrogram, log-scaled and normalized.
    """

    def __init__(self, filter_path: str):
        """
        Args:
            filter_path: Path to mel_filters.npy (shape: [201, 128])
        """
        self.filters = np.load(filter_path)  # (201, 128)

    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        """
        Extract mel spectrogram from audio waveform.

        Args:
            audio: 1D float32 waveform at 16kHz
            dtype: Output dtype (float32 for RKNN)

        Returns:
            (128, T) mel spectrogram where T = len(audio)//160 + 1
        """
        stft = _stft_numpy(audio, n_fft=400, hop_length=160, center=True)
        magnitudes = np.abs(stft) ** 2
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # Frame alignment: discard extra frame from center=True padding
        # (matches Qwen3-ASR official implementation)
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        return log_spec.astype(dtype)
