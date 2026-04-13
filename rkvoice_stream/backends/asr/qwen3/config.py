"""Configuration constants and default parameters."""

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]

# Default engine parameters
DEFAULT_CONFIG = {
    # Platform
    "platform": "rk3576",

    # RKLLM decoder parameters
    "max_context_len": 1024,    # Maximum KV cache context length
    "max_new_tokens": 128,      # Maximum tokens to generate per run (5s audio ≈ 50 tokens max)
    "top_k": 1,                 # Top-K sampling (1 = greedy)
    "top_p": 1.0,               # Nucleus sampling threshold
    "temperature": 1.0,         # Sampling temperature (1.0 with top_k=1 = greedy)
    "repeat_penalty": 1.15,      # Repetition penalty (>1.0 suppresses repetition, important for W4A16)
    "frequency_penalty": 0.0,   # Frequency penalty
    "presence_penalty": 0.0,    # Presence penalty

    # CPU affinity
    "enabled_cpus": 2,          # Number of CPU cores for RKLLM (2 = big cores only, recommended)

    # Streaming parameters
    "chunk_size": 5.0,          # Audio chunk size in seconds (5s optimal for streaming)
    "memory_num": 2,            # Number of recent chunks to keep in sliding window
    "unfixed_chunks": 0,        # First N chunks without prefix (0 = always use prefix, recommended for sliding window)
    "rollback_tokens": 0,       # Rollback tokens (0 = no rollback, recommended for sliding window)

    # VAD parameters
    "vad_threshold": 0.5,       # Speech detection threshold (0-1)
    "vad_min_silence": 0.5,     # Seconds of silence before speech ends
    "vad_min_speech": 0.25,     # Minimum speech duration (seconds)
    "vad_max_speech": 30.0,     # Maximum speech segment before force-split

    # Language and context
    "language": "Chinese",      # Default language (None = auto-detect)
    "context": "",              # Context hint for the model

    # Prompt optimization
    "compact_suffix": True,     # Use compact suffix (saves ~120ms per chunk)
                                # Full: "数字用0123456789，语音转录："
                                # Compact: "转录："

    # Encoder
    "encoder_mel_frames": 3000, # Static encoder mel frames (30s × 100 frames/s)
    "embed_dim": 1024,          # Embedding dimension
    "embed_flash": 1,           # Use flash embedding in RKLLM
}

# Encoder static shape constants
SAMPLE_RATE = 16000
MEL_FRAMES_30S = 3000          # 30s at 100 frames/s
ENCODER_TOKENS_30S = 390       # Output tokens for 30s input
MAX_AUDIO_SECONDS = 30.0       # Max chunk for static encoder

# CPU mask mapping for RK3576
CPU_MASKS = {
    2: 0xC0,   # CPU6, CPU7 (A72 big cores)
    4: 0xF0,   # CPU4-7
    1: 0x80,   # CPU7 only
    8: 0xFF,   # All cores
}
