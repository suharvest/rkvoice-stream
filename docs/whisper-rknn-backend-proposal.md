# Design proposal: `whisper_rknn` ASR backend

Status: **proposal / design only** (no implementation shipped with this change).
Author intent: give rkvoice-stream a broad-language offline ASR option on the
Rockchip NPU, complementing the existing `sensevoice_rknn` (fast zh/en) backend.

## 1. Motivation

rkvoice-stream currently has four ASR backends: `qwen3_asr_rk`,
`paraformer_rknn`, `sensevoice_rknn`, and the two `*_sherpa` CPU fallbacks.
None of them is Whisper. We have separately verified that **Whisper on RKNN**
(from the official `rockchip-linux/rknn_model_zoo` C++ example — encoder +
decoder as two `.rknn` graphs, no PyTorch at inference time) runs on-device on
both **rk3588** and **rv1126b**. Adding it as a first-class backend buys:

- **Broad language coverage** — Whisper covers ~99 languages; SenseVoice-small
  is effectively zh/yue/en/ja/ko. For anything outside that set, Whisper is the
  only on-device option we have.
- **An alternative rk3588 ASR** — SenseVoice on rk3588 needs the fiddly
  scaling-fp16 encoder (w4a16 is toolkit-rejected there). Whisper is a clean
  second option when that path is inconvenient.

**When to prefer which** (write this into the backend selection docs):

| Need | Backend |
|------|---------|
| Chinese / English, lowest latency, streaming partials | `sensevoice_rknn` (zh/en) or `qwen3_asr_rk` |
| Wide language set, offline whole-utterance | `whisper_rknn` |
| CPU-only fallback (no NPU) | `*_sherpa` |

## 2. Model layout & naming

Whisper is **two graphs**: an encoder (mel → audio features) and an
autoregressive decoder (tokens + encoder features → next token). Both are
frozen to fixed shapes for RKNN (no dynamic dims), exactly as SenseVoice is
frozen to `T_FIXED`.

Follow the existing per-SoC `.rknn` naming convention (mirrors
`sense-voice-encoder.<platform>.<dtype>.rknn`):

```
<WHISPER_RKNN_MODEL_DIR>/
  whisper-encoder.<size>.<platform>.<dtype>.rknn   # e.g. whisper-encoder.base.rv1126b.fp16.rknn
  whisper-decoder.<size>.<platform>.<dtype>.rknn
  whisper-mel-filters.npy                          # 80- (or 128-) bin mel filterbank
  whisper-vocab.json / tokenizer assets            # token id ↔ piece
```

`<size>` ∈ {`tiny`, `base`, `small`} (larger sizes exceed a 2 GB device's NPU
working set). `<platform>` reuses `RK_PLATFORM` (`rk3576` / `rk3588` /
`rv1126b`). Per-platform dtype is chosen at conversion time the same way
SenseVoice's is — see §5.

Model resolution mirrors `SenseVoiceRKNNBackend._resolve_model_path`: honour an
explicit `WHISPER_RKNN_ENCODER` / `WHISPER_RKNN_DECODER` override, else glob
`whisper-{encoder,decoder}.*.<platform>.*.rknn`.

## 3. Front end (mel)

Whisper's front end is a log-Mel spectrogram (80 bins for tiny/base/small;
128 for large-v3), 16 kHz, 25 ms window / 10 ms hop, padded/truncated to the
encoder's fixed frame count (Whisper uses 30 s = 3000 frames; a shortened
fixed window can be frozen to cut latency on edge). Two viable sources, both
already precedented in this repo:

- **`kaldi_native_fbank`** — already a dependency of `sensevoice_rknn`. Whisper
  uses a slightly different mel (Slaney-normalised, its own filterbank + log10
  + clamp), so we ship the exact Whisper filterbank as `whisper-mel-filters.npy`
  and apply it, rather than relying on kaldi's built-in mel.
- **Reuse `qwen3/mel.py`'s `MelExtractor`** — the Qwen3-ASR backend already
  carries a numpy mel extractor driven by a `mel_filters.npy`; Whisper's mel is
  the same shape of computation. Preferred: factor that extractor into a small
  shared `backends/asr/_mel.py` and parameterise the filterbank + normalisation.

Decision to record: **reuse the Qwen3 mel path**, generalised, to avoid a third
independent mel implementation.

## 4. Backend interface (stub)

Aligns with `rkvoice_stream.engine.asr.ASRBackend`. Offline whole-utterance, so
it opts into `supports_offline_streaming = True` and gets the generic
`OfflineAccumulateStream` (accumulate → transcribe on finalize) for free — no
per-backend stream code, endpointing delegated to the OVS server-side VAD, same
as SenseVoice.

```python
# rkvoice_stream/backends/asr/whisper_rknn.py  (PROPOSED — stub, not wired)
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import (
    ASRBackend, ASRCapability, TranscriptionResult,
)

# Whisper encoder frozen frame count (30 s * 100 fps = 3000; a shorter fixed
# window may be frozen to cut edge latency — must match conversion time).
ENC_FRAMES = 3000
N_MELS = 80


class WhisperRKNNBackend(ASRBackend):
    """Whisper (encoder+decoder) offline ASR on the Rockchip NPU via RKNNLite.

    Two RKNN graphs: encoder (mel -> audio features, one pass) and an
    autoregressive decoder (greedy/beam token loop). Broad language coverage;
    prefer sensevoice_rknn for low-latency zh/en. Verified on rk3588 + rv1126b.
    """

    supports_offline_streaming = True  # -> OfflineAccumulateStream, STREAMING cap

    def __init__(self) -> None:
        self._enc = None          # RKNNLite encoder
        self._dec = None          # RKNNLite decoder
        self._mel = None          # shared mel extractor (§3)
        self._tok = None          # tokenizer
        self._ready = False

    @property
    def name(self) -> str:
        return "whisper_rknn"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._enc is not None and self._dec is not None

    def preload(self) -> None:
        from rknnlite.api import RKNNLite
        from rkvoice_stream.platform import init_runtime_for_platform

        platform = os.environ.get("RK_PLATFORM", "rk3576").lower()
        enc_path, dec_path = self._resolve_model_paths()

        self._enc = RKNNLite(verbose=False)
        assert self._enc.load_rknn(enc_path) == 0, enc_path
        # Platform-aware init: single-core rv1126b -> maskless init_runtime();
        # multi-core rk3576/rk3588 -> core_mask (pin enc/dec to different cores
        # via WHISPER_RKNN_ENC_CORE / _DEC_CORE, cf. paraformer_rknn).
        assert init_runtime_for_platform(
            self._enc, platform=platform,
            core_mask=os.environ.get("WHISPER_RKNN_ENC_CORE", "NPU_CORE_0"),
        ) == 0

        self._dec = RKNNLite(verbose=False)
        assert self._dec.load_rknn(dec_path) == 0, dec_path
        assert init_runtime_for_platform(
            self._dec, platform=platform,
            core_mask=os.environ.get("WHISPER_RKNN_DEC_CORE", "NPU_CORE_0"),
        ) == 0

        self._mel = self._load_mel()        # §3 shared mel extractor
        self._tok = self._load_tokenizer()  # whisper-vocab.json
        self._ready = True

    def unload(self) -> None:
        for m in (self._enc, self._dec):
            if m is not None:
                try:
                    m.release()
                except Exception:
                    pass
        self._enc = self._dec = None
        self._ready = False

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        audio = self._decode_audio(audio_bytes)          # reuse sensevoice helper
        return self.transcribe_array(audio, language)

    def transcribe_array(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        mel = self._mel(audio)                            # (N_MELS, ENC_FRAMES), padded
        enc_out = self._enc.inference(inputs=[mel[None]])[0]
        text = self._decode_loop(enc_out, language)       # greedy AR token loop (§6)
        return TranscriptionResult(text=text, language=None)

    # --- helpers (stubs) ---------------------------------------------------
    def _resolve_model_paths(self) -> tuple[str, str]: ...   # glob per §2
    def _load_mel(self): ...                                 # §3
    def _load_tokenizer(self): ...
    def _decode_loop(self, enc_out: np.ndarray, language: str) -> str: ...
    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray: ...  # cf. sensevoice
```

Registration (one line in `engine/asr.py::create_asr_backend`):

```python
elif backend_name == "whisper_rknn":
    from rkvoice_stream.backends.asr.whisper_rknn import WhisperRKNNBackend
    return WhisperRKNNBackend()
```

## 5. Environment variables

| Var | Meaning |
|-----|---------|
| `RK_PLATFORM` | `rk3576` / `rk3588` / `rv1126b` — selects `.rknn` + init convention |
| `WHISPER_RKNN_MODEL_DIR` | model + asset directory (default `/opt/asr/whisper-rknn`) |
| `WHISPER_RKNN_ENCODER` / `WHISPER_RKNN_DECODER` | explicit `.rknn` overrides |
| `WHISPER_RKNN_ENC_CORE` / `WHISPER_RKNN_DEC_CORE` | NPU core mask on multi-core parts (ignored on single-core rv1126b) |
| `WHISPER_RKNN_SIZE` | `tiny` / `base` / `small` when several are present |

Per-platform dtype (freeze at conversion; same rationale as SenseVoice §
docstring in `sensevoice_rknn.py`): start with **fp16** on rv1126b/rk3576;
rk3588 fp16 may need the SenseVoice-style attention scaling if Chinese
activations overflow — validate on-device and record the chosen dtype in the
filename (`<dtype>` field).

## 6. Decoder loop notes

- Greedy CTC does **not** apply — Whisper's decoder is autoregressive. The loop
  is: prime with the forced-tokens prefix (`<|startoftranscript|>`,
  language tag, `<|transcribe|>`, `<|notimestamps|>`), then iteratively feed the
  running token sequence + encoder features to the decoder `.rknn`, take argmax
  (greedy) or a small beam, append, stop on `<|endoftext|>` or a max-token cap.
- RKNN has no KV-cache primitive; the reference `rknn_model_zoo` decoder is
  frozen to a fixed max decode length and re-runs the whole decoder graph per
  step. Cap `max_new_tokens` (e.g. 128) to bound latency — this is the main
  cost driver and the reason Whisper is offline-only here, not streaming.
- Language: `language="auto"` can use Whisper's own language-detection first
  token; an explicit tag skips it (faster, deterministic).

## 7. Scope / non-goals

- No streaming partials (autoregressive + no KV cache → too slow per hop). The
  offline→streaming adapter gives whole-utterance "streaming" only.
- No timestamps in v1 (`<|notimestamps|>`).
- Conversion tooling (ONNX/PyTorch → two `.rknn`) is **out of scope** for this
  runtime repo — it belongs with the model-conversion project, consuming the
  official `rknn_model_zoo` Whisper export. This proposal covers only the
  runtime backend contract.
```
