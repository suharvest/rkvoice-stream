# SenseVoice-small → RKNN (RK3576 / RK3588)

Reproduction guide for the `sensevoice_rknn` ASR backend
(`rkvoice_stream/backends/asr/sensevoice_rknn.py`). SenseVoice-small is an
**encoder + CTC** model — a single forward over LFR features yields
`[1, T, 25055]` CTC logits (no decoder, no CIF). Validated on real RK3576 NPU
(fp16, no overflow): English byte-identical to the FP32 ONNX reference, Chinese
matches modulo 1–2 chars.

## Source artifacts

- **Encoder ONNX**: `lovemefan/SenseVoice-onnx` → `sense-voice-encoder.onnx`
  (inputs `speech[1,T,560]` f32 + `speech_lengths[1]` i64; output `[1,T,25055]`).
- **Decode assets** (from the sherpa SenseVoice export
  `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`):
  - `am.mvn` — global CMVN (560-dim `add` shift + `scale`).
  - `embedding.npy` — `(16, 560)` prompt-embedding table.
  - `chn_jpn_yue_eng_ko_spectok.bpe.model` — sentencepiece tokenizer (25055).

## Conversion pipeline (x86 host with rknn-toolkit2 2.2.0, Python 3.10)

```bash
# 1) freeze the encoder to a fixed sequence length (RKNN has no dynamic dims).
#    speech_lengths is folded to a constant (= T_FIXED); padded frames are masked.
python sv_fix_shape.py sense-voice-encoder.onnx sense-voice-encoder.fixed.onnx 344

# 2) convert per SoC (run once for each target).
python convert_sensevoice_rknn.py --onnx sense-voice-encoder.fixed.onnx \
    --out sense-voice-encoder.rk3576.fp16.rknn --target rk3576 --precision fp16 --t-fixed 344
python convert_sensevoice_rknn.py --onnx sense-voice-encoder.fixed.onnx \
    --out sense-voice-encoder.rk3588.fp16.rknn --target rk3588 --precision fp16 --t-fixed 344
```

fp16 is clean on real RK3576 silicon (no overflow). If a future SoC/model
overflows, fall back to `--precision int8 --dataset <calib.txt>` (needs a
representative dataset of LFR features).

## Deployment layout

Place in `SENSEVOICE_RKNN_MODEL_DIR` (default `/opt/asr/sensevoice-rknn`):

```
sense-voice-encoder.rk3576.fp16.rknn   # selected by RK_PLATFORM=rk3576
sense-voice-encoder.rk3588.fp16.rknn   # selected by RK_PLATFORM=rk3588
am.mvn
embedding.npy
chn_jpn_yue_eng_ko_spectok.bpe.model
```

Select via `ASR_BACKEND=sensevoice_rknn` + `RK_PLATFORM={rk3576,rk3588}`.

## Decode contract (must match the backend exactly)

The front end + decode are reproduced in `sensevoice_rknn.py`; the values below
are the load-bearing constants.

| stage | detail |
|---|---|
| fbank | kaldi 80-dim, 16 kHz, `dither=0`, `window=hamming`, `snip_edges=True`, samples scaled `*32768` |
| LFR | `m=7, n=6` → 80×7 = **560**; left-pad `(m-1)//2` with first frame |
| CMVN | `(lfr + add) * scale` (both 560-dim, from `am.mvn`) |
| prompt prefix | 4 frames prepended: `[emb[LANG_IDS[lang]], emb[1], emb[2], emb[TEXTNORM[textnorm]]]` |
| fixed length | `T_FIXED = 344` (pad zeros / truncate); track `valid` = real frame count |
| logits | encoder output `[T_FIXED, 25055]`; decode only the first `valid` frames |
| CTC | greedy argmax → collapse repeats → drop blank (id **0**) |
| tokens | sentencepiece `id_to_piece`; `▁` → space; strip `<\|...\|>` prompt/emotion/event/itn tokens via regex |

Prompt table indices (`embedding.npy` rows): language `auto=0, zh=3, en=4,
yue=7, ja=11, ko=12`; textnorm `withitn=14, woitn=15`; row 1 = event, row 2 =
speech. Special token ids in the bpe vocab: `<|zh|>=24884 <|en|>=24885
<|yue|>=24888 <|ja|>=24892 <|ko|>=24896 <|withitn|>=25016 <|woitn|>=25017`.

## Notes / gotchas

- The PC `rknn-toolkit2` **simulator cannot run a `load_rknn` artifact**
  (`init_runtime: ... not support inference on the simulator`) — validate on real
  hardware via `rknnlite`, or keep the `RKNN` object from `load_onnx`+`build`.
- On the RK devices, probe/runtime containers need `--privileged --network host`
  (the default docker bridge has no egress; resolv points at Tailscale DNS).
- librknnrt on the validated RK3576 (cat-remote): 2.3.2, NPU driver 0.9.8.
