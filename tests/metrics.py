"""Reusable metrics for speech pipeline evaluation.

Zero external dependencies beyond numpy + stdlib.
Can be imported standalone: `from rk3576.tests.metrics import cer`
"""

from __future__ import annotations

import io
import re
import wave


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (character-level), two-row DP."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def normalize_text(text: str) -> str:
    """Strip punctuation and whitespace for fairer CER comparison."""
    # Remove Chinese and ASCII punctuation, whitespace
    text = re.sub(
        r'[\s，。！？、；：\u201c\u201d\u2018\u2019（）,.!?;:\'"()\[\]{}\-]',
        '',
        text,
    )
    return text.lower()


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (0.0 - 1.0+). Returns 0.0 for empty reference."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors (numpy arrays or lists)."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Return duration of a WAV bytestring in seconds."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def compute_rtf(inference_s: float, audio_duration_s: float) -> float:
    """Real-Time Factor: inference time / audio duration. Lower is better."""
    if audio_duration_s <= 0:
        return float('inf')
    return inference_s / audio_duration_s
