"""
Voice Activity Detection (VAD) module using Silero VAD via sherpa_onnx.

Provides a lightweight VAD wrapper for the streaming ASR pipeline:
- Silero VAD: ~1.5ms per 32ms frame on RK3576 CPU, RTF=0.043
- Detects speech start/end with configurable thresholds
- Accumulates speech audio for ASR processing

Requirements:
    - sherpa_onnx (already installed on RK3576)
    - silero_vad.onnx model file
"""

import numpy as np
from typing import Optional, Callable, List, Tuple

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512  # Silero VAD expects 512 samples (32ms at 16kHz)


class SileroVAD:
    """
    Silero VAD wrapper using sherpa_onnx.VoiceActivityDetector.
    
    Usage:
        vad = SileroVAD("/path/to/silero_vad.onnx")
        
        # Feed audio in any size chunks
        vad.feed(pcm_chunk)
        
        # Check for completed speech segments
        while vad.has_speech():
            audio, start_sec, duration = vad.pop_speech()
            # Process audio with ASR...
        
        # Check if currently in speech
        if vad.is_speech:
            ...
    """

    def __init__(self,
                 model_path: str,
                 threshold: float = 0.5,
                 min_silence_duration: float = 0.5,
                 min_speech_duration: float = 0.25,
                 max_speech_duration: float = 30.0,
                 num_threads: int = 1,
                 buffer_size_seconds: float = 120.0):
        """
        Args:
            model_path: Path to silero_vad.onnx
            threshold: Speech detection threshold (0-1). Lower = more sensitive.
            min_silence_duration: Seconds of silence before speech is considered ended.
            min_speech_duration: Minimum speech duration to be considered valid.
            max_speech_duration: Maximum speech segment duration (force-split).
            num_threads: CPU threads for ONNX inference.
            buffer_size_seconds: Internal buffer size for sherpa_onnx.
        """
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence_duration
        config.silero_vad.min_speech_duration = min_speech_duration
        config.silero_vad.max_speech_duration = max_speech_duration
        config.sample_rate = SAMPLE_RATE
        config.num_threads = num_threads

        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=buffer_size_seconds
        )
        self._buffer = np.zeros(0, dtype=np.float32)
        self._total_samples = 0  # Total samples fed (for timing)
        self._is_speech = False

    @property
    def is_speech(self) -> bool:
        """Whether the VAD currently detects speech activity."""
        return self._vad.is_speech_detected()

    def feed(self, pcm16k: np.ndarray):
        """
        Feed audio data to VAD. Processes in 512-sample frames.
        
        Args:
            pcm16k: 1D float32 array at 16kHz mono
        """
        x = np.asarray(pcm16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Append to internal frame buffer
        if self._buffer.shape[0] > 0:
            x = np.concatenate([self._buffer, x])

        # Process complete frames
        n_frames = len(x) // VAD_WINDOW_SIZE
        for i in range(n_frames):
            frame = x[i * VAD_WINDOW_SIZE:(i + 1) * VAD_WINDOW_SIZE]
            self._vad.accept_waveform(frame)
            self._total_samples += VAD_WINDOW_SIZE

        # Save remainder
        remainder = len(x) - n_frames * VAD_WINDOW_SIZE
        if remainder > 0:
            self._buffer = x[-remainder:]
        else:
            self._buffer = np.zeros(0, dtype=np.float32)

    def has_speech(self) -> bool:
        """Check if there are completed speech segments ready to retrieve."""
        return not self._vad.empty()

    def pop_speech(self) -> Tuple[np.ndarray, float, float]:
        """
        Pop one completed speech segment.
        
        Returns:
            (audio, start_sec, duration_sec) tuple:
                audio: float32 PCM at 16kHz
                start_sec: Start time in seconds from stream beginning
                duration_sec: Duration of the speech segment
        """
        seg = self._vad.front
        audio = np.array(seg.samples, dtype=np.float32)
        start_sec = seg.start / SAMPLE_RATE
        duration_sec = len(audio) / SAMPLE_RATE
        self._vad.pop()
        return audio, start_sec, duration_sec

    def pop_all_speech(self) -> List[Tuple[np.ndarray, float, float]]:
        """Pop all completed speech segments."""
        segments = []
        while self.has_speech():
            segments.append(self.pop_speech())
        return segments

    def flush(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        Flush any remaining buffered audio and pop all segments.
        Call this at end of stream to get the final speech segment.
        """
        # Feed remaining buffer samples (pad to 512 if needed)
        if self._buffer.shape[0] > 0:
            pad_len = VAD_WINDOW_SIZE - self._buffer.shape[0]
            if pad_len > 0:
                padded = np.pad(self._buffer, (0, pad_len))
            else:
                padded = self._buffer
            self._vad.accept_waveform(padded[:VAD_WINDOW_SIZE])
            self._buffer = np.zeros(0, dtype=np.float32)

        # Force flush the VAD internal state
        self._vad.flush()

        return self.pop_all_speech()

    @property
    def elapsed_seconds(self) -> float:
        """Total audio duration processed so far."""
        return self._total_samples / SAMPLE_RATE

    def reset(self):
        """Reset VAD state for a new stream."""
        self._vad.reset()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._total_samples = 0
