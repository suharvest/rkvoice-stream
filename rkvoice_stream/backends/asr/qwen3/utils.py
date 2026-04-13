"""Audio loading and text processing utilities."""

import os
import numpy as np
from .config import SAMPLE_RATE


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE,
               start_second: float = 0.0,
               duration: float = None) -> np.ndarray:
    """
    Load audio file and convert to 16kHz mono float32 PCM.
    
    Supports: wav, mp3, m4a, flac, ogg, and any format supported by
    soundfile or pydub (with ffmpeg).
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16000)
        start_second: Start time in seconds
        duration: Duration in seconds (None = full file)
        
    Returns:
        1D float32 numpy array in [-1, 1]
    """
    audio_path = str(audio_path)
    ext = os.path.splitext(audio_path)[1].lower()

    if ext in ('.wav', '.flac', '.ogg'):
        try:
            import soundfile as sf
            # Get file sample rate first for correct start offset
            info = sf.info(audio_path)
            file_sr = info.samplerate
            start_frame = int(start_second * file_sr) if start_second > 0 else 0
            audio, file_sr = sf.read(audio_path, dtype="float32",
                                     start=start_frame)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=file_sr,
                                         target_sr=sample_rate)
            if duration:
                max_samples = int(duration * sample_rate)
                audio = audio[:max_samples]
            return audio.astype(np.float32)
        except Exception:
            pass

    # Fallback: pydub (requires ffmpeg)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(audio_path)
        if start_second > 0:
            seg = seg[int(start_second * 1000):]
        if duration:
            seg = seg[:int(duration * 1000)]
        seg = seg.set_channels(1).set_frame_rate(sample_rate)
        max_val = float(1 << (seg.sample_width * 8 - 1))
        audio = np.array(seg.get_array_of_samples()) / max_val
        return audio.astype(np.float32)
    except ImportError:
        pass

    # Last resort: librosa
    import librosa
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True,
                            offset=start_second, duration=duration)
    return audio.astype(np.float32)


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """
    Detect and remove excessive repetitions in transcribed text.
    
    From Qwen3-ASR official utils.
    """
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i + count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while (end_index + k <= n
                           and s[end_index:end_index + k] == pattern):
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(
                        s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def apply_itn(text: str) -> str:
    """
    Apply Inverse Text Normalization (Chinese number conversion etc.).
    
    Tries to import chinese_itn from various locations. Returns original
    text if ITN module is not available.
    """
    try:
        import sys
        # Try direct import
        try:
            from qwen_asr_gguf import chinese_itn
            return chinese_itn.chinese_to_num(text)
        except ImportError:
            pass

        # Try common locations on RK3576
        for path in [
            "/home/qztest/qwen3asr_rknn/qwen_asr_gguf",
            "/home/qztest/qwen3asr_rknn",
        ]:
            if path not in sys.path and os.path.isdir(path):
                sys.path.insert(0, path)
                try:
                    import chinese_itn
                    return chinese_itn.chinese_to_num(text)
                except ImportError:
                    sys.path.remove(path)
    except Exception:
        pass

    return text


def parse_asr_output(raw: str, user_language: str = None):
    """
    Parse raw LLM output into (language, text).
    
    The model may output: "language Chinese<asr_text>transcribed text..."
    or just plain text if language was forced in prompt.
    
    Args:
        raw: Raw decoded text from LLM
        user_language: Forced language (if provided, raw is treated as plain text)
        
    Returns:
        (language: str, text: str)
    """
    if not raw:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        return user_language, s

    ASR_TEXT_TAG = "<asr_text>"
    LANG_PREFIX = "language "

    if ASR_TEXT_TAG in s:
        meta, text = s.split(ASR_TEXT_TAG, 1)
        if "language none" in meta.lower():
            return "", text.strip() if text.strip() else ""
        lang = ""
        for line in meta.splitlines():
            line = line.strip()
            if line.lower().startswith(LANG_PREFIX):
                val = line[len(LANG_PREFIX):].strip()
                if val:
                    lang = val[:1].upper() + val[1:].lower()
                break
        return lang, text.strip()
    else:
        return "", s.strip()
