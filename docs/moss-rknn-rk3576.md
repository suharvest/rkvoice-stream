# MOSS-TTS-Nano RK3576 Production Port

Status: ONNX Runtime streaming baseline remains the production correctness fallback on RK3576. RKNN hybrid prefill runners are available only as explicit experiments: the original `ln2 + MLP` path improved TTFA but failed ASR roundtrip quality, the `fc_out_only` path has better short sampler parity but is slower than full ORT at service level, and the current best coarse prefill path `ln1_cattn` passes the 20-frame service streaming gate but is still slightly slower than full ORT under the short Junhao production prompt. No hybrid path may be promoted to the default production profile until it beats the canonical ORT service profile and passes the same ASR roundtrip quality gate.

Official RKLLM custom export is also experimental for MOSS. The current
embed-only folded `.rkllm` loads on RK3576, runs prefill/decode, and can produce
non-silent audio through the existing ORT sampler/codec chain, but hidden-state
parity against the ONNX baseline fails badly (`rel_l2≈1.32`, cosine `≈0.13` for
both prefill and decode on the short s8 probe). Treat RKLLM smoke audio as a
runtime sanity check only. It cannot be promoted until
`compare_moss_rkllm_hidden_runtime.py` passes the ONNX hidden parity gate on
RK3576.

Community/official RKLLM feedback checked on 2026-05-24 points to the same
direction: embedding input has had runtime fixes in official releases, and
community users use it mainly for multimodal LLaMA/Qwen-style models. That is
not the same as arbitrary GPT2/MOSS block parity. The official custom demo is
LLaMA/MiniCPM-shaped (`RMSNorm`, qkv projection, RoPE, gated FFN). MOSS is
LayerNorm+bias/GPT2 Conv1D-style, so a successful custom export/load is not a
production accuracy proof. Local RK3576 probes also rule out the obvious fixes:
`RKLLM_EMBED_FLASH=0` leaves the hidden mismatch unchanged, token input still
fails hidden parity (`rel_l2≈1.29`, cosine `≈0.17`), and a WSL2 HF RMSNorm
variant remains far from RKLLM hidden (`rel_l2≈1.32`, cosine `≈0.13`) while the
original HF scaffold still matches ONNX. If RKLLM remains the preferred NPU
path, use `models/tts/moss/package_moss_rkllm_reproducer.py` to package the
current minimal official reproducer: fixed input IDs/embeddings, ONNX/HF hidden
output, RKLLM hidden output, token-input and `embed_flash=0` negative probes,
and an upstream issue draft. The verified local package is
`/tmp/moss-rkllm-upstream-reproducer`. In parallel, RKNN/RKMatMul NPU islands
must be validated only at deterministic hidden/logit boundaries; ORT is a
golden checker, not the target runtime.

## Production Contract

The RKNN runtime profile is `configs/rk3576-moss-rknn-stream.yaml` and backend
name is `moss_rknn`. This is a production-candidate profile, not the current
default production profile. It sets `require_production_default: 1`, so the
backend will load only a manifest whose `quality_status.production_default=true`
is backed by full production evidence.

The current RK3576 service fallback profile is `configs/rk3576-moss-ort-stream.yaml` and backend name is `moss_ort`.

Current verified ORT production settings:

```yaml
tts:
  backend: moss_ort
  require_backend: 1
  model_dir: /opt/tts/models/moss-tts-nano-onnx
  voice: Junhao
  seed: 314
  threads: 6
  prefill_threads: 8
  decode_threads: 5
  codec_threads: 5
  prefill_seq: 0
  max_new_frames: 20
  codec_streaming: 1
  codec_batch_frames: 3
  cache_voice_prefix: 0
  manifest: moss-ort-manifest.json
  warmup_text: 你好
  allow_deterministic_fallback: 0
```

The current RK3576 hybrid experimental profile is `configs/rk3576-moss-hybrid-rknn-stream.yaml`, also using backend name `moss_ort` with the hybrid prefill flags enabled. Hybrid strict mode must be enabled for verification so a missing or invalid RKNN island does not silently fall back to full ORT. The config file still records the older `fc_out_only` diagnostic route; the current best verified NPU prefill route is the explicit service-profile override `MOSS_ORT_HYBRID_SPLIT=ln1_cattn` with all 12 layers:

```yaml
tts:
  backend: moss_ort
  require_backend: 1
  model_dir: /opt/tts/models/moss-tts-nano-onnx
  voice: Junhao
  threads: 6
  codec_streaming: 1
  cache_voice_prefix: 0
  warmup_text: 你好
  hybrid_rknn: 1
  hybrid_strict: 1
  hybrid_seq_len: 320
  hybrid_split: ln1_cattn
  hybrid_layers: all
  hybrid_dir: /opt/tts/models/moss-tts-nano-hybrid-rknn
  hybrid_rknn_dir: /opt/tts/models/moss-tts-nano-hybrid-ln1-cattn-rknn
```

Current production baseline should use ONNX Runtime for correctness and streamability while RKNN subgraphs are isolated. Required ORT baseline layout:

```text
/opt/tts/models/moss-tts-nano-onnx/
  tokenizer.model
  tts_browser_onnx_meta.json
  codec_browser_onnx_meta.json
  moss_tts_prefill.onnx
  moss_tts_decode_step.onnx
  moss_tts_local_fixed_sampled_frame.onnx
  moss_tts_global_shared.data
  moss_tts_local_shared.data
  moss_audio_tokenizer_decode_full.onnx
  moss_audio_tokenizer_decode_step.onnx
  moss_audio_tokenizer_decode_shared.data
```

Low-latency dialogue path:

```text
prefill global model -> local fixed-frame sampler -> codec streaming decode_step
```

This produces 3840-sample stereo chunks at 48 kHz, which is 80 ms of audio per chunk. Preload is intentionally excluded from TTFA; production must preload sessions at service start. `moss_audio_tokenizer_decode_step.onnx` is required for production streaming. If it is missing, the backend falls back to repeated full codec decode and the per-frame codec cost grows with generated length.

The dialogue path must consume `TTSBackend.synthesize_stream()` directly.
`/tts/stream`, `DialogueOrchestrator.process_turn_pcm()`, and `/dialogue`
all reject non-streaming TTS backends instead of falling back to full
`synthesize()` WAV generation. `/dialogue` sends a 4-byte little-endian
sample-rate header followed by backend-native int16 PCM chunks as soon as they
are decoded. Streaming worker failures must propagate to the service layer:
`/tts/stream` aborts the response generator and `/dialogue` sends a JSON error
instead of silently treating truncated audio as success. `/health` exposes
`streaming_tts=true` for `moss_ort` and
`moss_rknn` so clients can reject non-streaming profiles. For `moss_ort`,
`/health.tts_info` also exposes the loaded voice/seed/thread profile and
whether the ORT manifest was hash-validated. `moss_ort` additionally reports
`streaming_stats` (`requests`, `completed`, `errors`, `active`, `chunks`, and
the last stream error) so production monitors can distinguish a healthy process
from a degraded streaming path. The service streaming verifier reads `/health`
before and after `/tts/stream` plus `/dialogue`; when production runtime gates
are enabled, it fails if `streaming_stats.errors` increases or
`streaming_stats.active` is non-zero after the run.

Required deployment layout:

```text
/opt/tts/models/moss-tts-nano-onnx/
  moss-ort-manifest.json
  tokenizer.model
  tts_browser_onnx_meta.json
  codec_browser_onnx_meta.json
  moss_tts_prefill.onnx
  moss_tts_decode_step.onnx
  moss_tts_local_fixed_sampled_frame.onnx
  moss_tts_global_shared.data
  moss_tts_local_shared.data
  moss_audio_tokenizer_decode_full.onnx
  moss_audio_tokenizer_decode_step.onnx
  moss_audio_tokenizer_decode_shared.data

/opt/tts/models/moss-tts-nano-rknn/
  moss-rknn-manifest.json
  tokenizer.model
  tts_browser_onnx_meta.json
  codec_browser_onnx_meta.json
  moss_tts_prefill.s32.fp16.rk3576.rknn
  moss_tts_prefill.s64.fp16.rk3576.rknn
  moss_tts_prefill.s128.fp16.rk3576.rknn
  moss_tts_prefill.s256.fp16.rk3576.rknn
  moss_tts_decode_step.p1.fp16.rk3576.rknn
  moss_tts_decode_step.p32.fp16.rk3576.rknn
  moss_tts_decode_step.p64.fp16.rk3576.rknn
  moss_tts_decode_step.p128.fp16.rk3576.rknn
  moss_tts_decode_step.p256.fp16.rk3576.rknn
  moss_tts_decode_step.p512.fp16.rk3576.rknn
  moss_tts_local_fixed_sampled_frame.fp16.rk3576.rknn
  codec_decode_step.f1.fp16.rk3576.rknn
  codec_decode_step.f4.fp16.rk3576.rknn
  codec_decode_step.f8.fp16.rk3576.rknn
/opt/rkvoice-workers/moss_rknn_worker
```

`moss-rknn-manifest.json` must include `quality_status`. New manifests written
by `write_moss_rknn_manifest.py` default to `production_default=false`. The
`moss_rknn` backend rejects a manifest that claims `production_default=true`
unless it also carries `production_evidence.passed=true` with all four current
production checks passing: artifact manifest, service streaming, backend-stage,
and ASR roundtrip quality. When `MOSS_RKNN_REQUIRE_PRODUCTION_DEFAULT=1` is set
by the RKNN production-candidate profile, the backend also rejects manifests
that do not claim `production_default=true`. This prevents sampler/preload RKNN
probes from being promoted by changing only the config file or manifest label.

`moss-ort-manifest.json` is mandatory for the RK3576 production fallback profile in `configs/rk3576-moss-ort-stream.yaml`. Generate it with:

```bash
python models/tts/moss/write_moss_ort_manifest.py \
  --model-dir /opt/tts/models/moss-tts-nano-onnx \
  --target rk3576 \
  --voice Junhao \
  --seed 314

python models/tts/moss/verify_moss_ort_artifacts.py \
  --model-dir /opt/tts/models/moss-tts-nano-onnx

python models/tts/moss/verify_moss_ort_config.py \
  --config configs/rk3576-moss-ort-stream.yaml \
  --require-model-dir \
  --validate-artifacts \
  --json-out /tmp/moss_ort_config_artifacts_contract.json
```

The `moss_ort` backend validates the manifest when `MOSS_ORT_MANIFEST` is set or when `moss-ort-manifest.json` exists. Every required artifact must record `path`, `size_bytes`, and `sha256`; preload fails if any value differs. The ORT manifest also marks `streaming_required=true`, so `moss_audio_tokenizer_decode_step.onnx` is a required production artifact. `verify_moss_ort_config.py` is the config contract gate: it rejects drift from the production profile (`Junhao`, seed `314`, base `threads=6`, `prefill_threads=8`, `decode_threads=5`, `codec_threads=5`, `max_new_frames=20`, `codec_batch_frames=3`, `warmup_text=你好`, stream codec enabled, prefix cache disabled, deterministic fallback disabled) and can also validate the artifact manifest from the configured model directory.

The same config gate also rejects experimental fields in the production ORT
profile: `hybrid_rknn`, `hybrid_strict`, `hybrid_seq_len`, `hybrid_split`,
`hybrid_layers`, `hybrid_dir`, `hybrid_rknn_dir`, `hybrid_manifest`, and
`codec_async` must be absent or explicitly disabled. `load_full_codec` must
also be absent or disabled in the low-latency production profile because the
streaming codec path does not need the full-codec session. These paths are
useful for lab probes, but the latest RK3576 evidence shows they are not
production-safe defaults.

`codec_async` is still mapped by `_apply_tts_env()` for controlled lab profiles,
but checked-in production config tests require `configs/rk3576-moss-ort-stream.yaml`
to omit or disable it. Do not use that mapping to bypass the production
verifier.

The service-level verifier also requires the live `/health.tts_info` runtime
profile to prove the same contract: `profile.codec_async=false`,
`profile.codec_full_loaded=false`, `profile.cache_voice_prefix=false`, and
`hybrid.enabled=false`. This catches environment-variable contamination even
when the YAML file is correct.

`moss-rknn-manifest.json` is mandatory. Every required artifact must record `path`, `size_bytes`, and `sha256`; preload fails if any value differs.

Hybrid deployment layout:

```text
/opt/tts/models/moss-tts-nano-hybrid-rknn/
  moss-hybrid-manifest.json
  moss_embedding_prefix.s320.onnx
  moss_final_norm.s320.onnx
  moss_block0_attn_residual.s320.onnx
  moss_block0_ln2_mlp.s320.fp16.rk3576.rknn
  ...
  moss_block11_attn_residual.s320.onnx
  moss_block11_ln2_mlp.s320.fp16.rk3576.rknn
```

`moss-hybrid-manifest.json` is required for the experimental hybrid profile when `hybrid_manifest` is set. Generate and verify it with:

```bash
python models/tts/moss/write_moss_hybrid_manifest.py \
  --artifact-dir /opt/tts/models/moss-tts-nano-hybrid-rknn \
  --target rk3576 \
  --seq-len 320 \
  --split ln2_mlp \
  --layers all

python models/tts/moss/verify_moss_hybrid_artifacts.py \
  --artifact-dir /opt/tts/models/moss-tts-nano-hybrid-rknn \
  --target rk3576 \
  --seq-len 320 \
  --split ln2_mlp \
  --layers all
```

The hybrid manifest records `path`, `size_bytes`, and `sha256` for every ORT attention slice and RKNN MLP island. The backend rejects a manifest that claims `quality_status.production_default=true`; hybrid cannot be promoted until the same ASR roundtrip quality gate as full ORT passes.

For split experiments such as the current `fc_out_only` candidate, the manifest
must also record the split type and exact RKNN layer set. Generate and validate
the same candidate with:

```bash
python models/tts/moss/write_moss_hybrid_manifest.py \
  --artifact-dir /opt/tts/models/moss-tts-nano-hybrid-rknn \
  --rknn-dir /opt/tts/models/moss-tts-nano-hybrid-fc-split-rknn \
  --out moss-hybrid-fc-out-manifest.json \
  --target rk3576 \
  --seq-len 320 \
  --split fc_out_only \
  --layers 0,1,4,5,6

python models/tts/moss/verify_moss_hybrid_artifacts.py \
  --artifact-dir /opt/tts/models/moss-tts-nano-hybrid-rknn \
  --rknn-dir /opt/tts/models/moss-tts-nano-hybrid-fc-split-rknn \
  --manifest moss-hybrid-fc-out-manifest.json \
  --target rk3576 \
  --seq-len 320 \
  --split fc_out_only \
  --layers 0,1,4,5,6
```

When `MOSS_ORT_HYBRID_MANIFEST` is set, `moss_ort` validates the manifest
against the requested `hybrid_split` and `hybrid_layers`; a split or layer drift
is a preload failure, not a warning.

## RKNN Build Rules

- Use the Jetson playbook phases T0-T6 as stop criteria, not as loose guidance.
- Export ONNX with fixed shapes for RKNN. Avoid dynamic axes; build multiple fixed buckets instead.
- Run `onnxsim` and the MOSS codec If-rank / bool-control-flow surgery before RKNN conversion.
- Start FP16 first. INT8 is allowed only after ASR roundtrip quality passes on FP16 and the calibration set is recorded.
- Verify each RKNN on the RK3576 device with `rknnlite`, not only toolkit simulator.
- Keep RKLLM/ASR coexistence in mind; if ASR and TTS share the NPU, serialize access or pin domains explicitly.

## Current NPU Route Decision

Target route remains high-performance streaming TTS, not an ORT-equivalent
fallback. The current priority order is:

1. Codec streaming RKNN, if the full cache-carrying decode-step can pass
   RK3576 `inference()` and parity. This is the cleanest latency target because
   it should not change text/sampler decisions and can reduce first-audio plus
   steady-state codec cost.
2. Sampler fused MLP / text-head RKNN islands. These have real-device parity
   and isolated speedups, but the total expected service gain is smaller because
   CPU/NPU handoff and sequential sampler dependencies can eat the win.
3. `ln1_cattn` prefill hybrid. It is the widest verified prefill RKNN route
   today, but the short Junhao production profile is still slightly slower than
   full ORT, so it is not the default route.
4. RKLLM only after hidden parity is fixed. Official RKLLM is faster at the raw
   prefill/decode stage, but current MOSS hidden-state parity failures make it
   unsafe for production audio quality even when smoke audio is non-silent.

Codec RKNN status as of the RK3576 `f1` probes:

- The original codec RKNN build failed in RKNN toolkit rule
  `merge_conv_channel_inner_perm`. Passing
  `disable_rules=["merge_conv_channel_inner_perm"]` builds
  `codec_decode_step.f1.fp16.rk3576.rknn`.
- Runtime then failed on unsupported CPU fallback op `Xor`. Graph inspection
  showed this was exactly `Xor(x, false)`, so `convert_moss_rknn.py` now rewrites
  it to `Identity(x)` before conversion.
- The Xor-fixed model builds, and the unsupported-op runtime error is gone, but
  RK3576 still SIGSEGVs during `inference()` with `pass-through=auto`, `none`,
  and `all`. This rules out input pass-through as the primary cause.
- Trying to crop codec outputs to `audio/audio_lengths` with RKNN `outputs=...`
  currently fails during toolkit `fold_constant` even at `optimization_level=0`,
  so the audio-only route is not yet a usable shortcut.
- Fixed-shape `onnxsim` is required for reliable codec crop probes. Without it,
  a `/rope/Add_output_0` crop hits the RKNN toolkit `fold_constant` bug; with
  `--codec-simplify`, the official converter can build the crop.
- Full codec still crashes even after `audio_codes/audio_code_lengths` are moved
  to CPU-side INT64 inputs. Crop bisection proves the first stable NPU boundary
  reaches `i95` (Q/K/V gather), while `i96` `Cast(attn_offset_0 -> INT64)` and
  `i97` `Cast(INT64 -> FLOAT)` expose RK3576 runtime SIGSEGV paths. The
  converter now rewrites input-side Cast-to-INT64 as INT64 input ABI and records
  this in `int64offset` codec artifact names plus `input_dtypes` metadata.
- The current codec route is therefore not full-graph RKNN. It is:
  `front RKNN through Q/K/V -> CPU RoPE/attention mask/attention -> suffix RKNN`.
  The first suffix island, `codec_suffix_layer0_outproj_ffn`, takes CPU attention
  output plus residual hidden and runs out-projection + norm2 + FFN on RK3576 in
  about `2.5 ms` with finite output.

Evidence:

- `docs/evidence/moss/wsl2-moss-codec-decode-step-graph-inspect-bool.json`
- `docs/evidence/moss/wsl2-moss-codec-f1-xorfix-manifest.json`
- `docs/evidence/moss/rk3576-moss-codec-f1-xorfix-runtime-probe.json`
- `docs/evidence/moss/rk3576-moss-codec-f1-xorfix-runtime-probe-none.json`
- `docs/evidence/moss/rk3576-moss-codec-f1-xorfix-runtime-probe-all.json`
- `docs/evidence/moss/wsl2-moss-codec-f1-xorfix-audio-only-build.log`
- `docs/evidence/moss/wsl2-moss-codec-f1-xorfix-audio-only-opt0-build.log`
- `docs/evidence/moss/wsl2-moss-codec-official-crop-i99-simplify-manifest.json`
- `docs/evidence/moss/rk3576-moss-codec-official-crop-i99-simplify-int64offset-runtime-probe-auto.json`
- `docs/evidence/moss/wsl2-moss-codec-suffix-layer0-outproj-ffn-build.json`
- `docs/evidence/moss/rk3576-moss-codec-suffix-layer0-outproj-ffn-runtime-probe-auto.json`

## Gates

Minimum production gates for RK3576:

```json
{
  "max_ttfa_ms": 1500,
  "max_stream_codec_ms": 170,
  "min_dialogue_binary_chunks": 7,
  "max_dialogue_payload_gap_ms": 1500,
  "max_avg_asr_cer": 0.5,
  "min_non_silent_rms": 0.02
}
```

The current ORT baseline is a production fallback for correctness and streamability, not the final performance target. The original Jetson target gates (`max_ttfa_ms=500`, `max_rtf=0.75`, low CER) still require RKNN or a verified prefix-cache/subgraph split.

Verification commands:

```bash
python models/tts/moss/audit_moss_release.py \
  --min-root-free-mb 512 \
  --json-out /tmp/moss_release_audit.json

python models/tts/moss/audit_moss_release.py \
  --validate-ort-artifacts \
  --json-out /tmp/moss_release_audit_ort_artifact_gate.json

# Optional RKNN acceleration preflight. This is not required for the current
# ORT production fallback, but it must pass before starting another large RKNN
# conversion/probe cycle or promoting RKNN artifacts.
python models/tts/moss/audit_moss_release.py \
  --require-rknn-workspace \
  --rknn-workspace /mnt/rknn-workspace/moss-rknn-workspace \
  --json-out /tmp/moss_release_audit_rknn_workspace_required.json

python models/tts/moss/run_moss_production_server.py \
  --dry-run \
  --python /home/cat/rknn-venv/bin/python \
  --host 0.0.0.0 \
  --port 8621 \
  --json-out /tmp/moss_production_server_dry_run.json

python models/tts/moss/run_moss_production_server.py \
  --dry-run \
  --min-root-free-mb 512 \
  --disk-path / \
  --json-out /tmp/moss_production_server_disk_gate.json

python models/tts/moss/run_moss_production_server.py \
  --dry-run \
  --require-rknn-workspace \
  --json-out /tmp/moss_production_server_rknn_workspace_required.json

python models/tts/moss/audit_moss_disk.py \
  --home-dir /home/cat \
  --min-candidate-mb 100 \
  --json-out /tmp/moss_disk_audit.json

# Optional cleanup after reviewing /tmp/moss_disk_audit.json. This only deletes
# reported /tmp candidates; it never deletes /home candidates.
python models/tts/moss/audit_moss_disk.py \
  --home-dir /home/cat \
  --min-candidate-mb 100 \
  --delete-tmp-candidates \
  --confirm-delete DELETE_MOSS_TMP_CANDIDATES \
  --json-out /tmp/moss_disk_cleanup.json

python models/tts/moss/verify_moss_production_profile.py \
  --model-dir /opt/tts/models/moss-tts-nano-onnx \
  --manifest moss-ort-manifest.json \
  --asr-model-dir /opt/asr/paraformer \
  --codec-batch-frames 3 \
  --min-dialogue-binary-chunks 7 \
  --min-backend-audio-frames 20 \
  --max-backend-ttfa-ms 1500 \
  --max-backend-prefill-ms 1200 \
  --json-out /tmp/moss_production_profile.json

python models/tts/moss/verify_moss_service_profile.py \
  --config configs/rk3576-moss-ort-stream.yaml \
  --port 8624 \
  --json-out /tmp/moss_service_profile_gate.json

python models/tts/moss/verify_moss_ort_stream.py \
  --model-dir /opt/tts/models/moss-tts-nano-onnx \
  --json-out /tmp/moss-ort-stream-verify.json

python models/tts/moss/verify_moss_ort_roundtrip.py \
  --model-dir /opt/tts/models/moss-tts-nano-onnx \
  --asr-model-dir /opt/asr/paraformer \
  --out-dir /tmp/moss_ort_roundtrip_verify \
  --json-out /tmp/moss_ort_roundtrip_verify.json

CONFIG=configs/rk3576-moss-ort-stream.yaml \
python -m rkvoice_stream.app.server

# Production entrypoint: audits the checked-in release contract before execing
# uvicorn with CONFIG=configs/rk3576-moss-ort-stream.yaml.
python models/tts/moss/run_moss_production_server.py \
  --python /home/cat/rknn-venv/bin/python \
  --host 0.0.0.0 \
  --port 8621

# Deployment preflight only. This must pass before using the production
# entrypoint above on a board image.
python models/tts/moss/run_moss_production_server.py \
  --dry-run \
  --validate-artifacts \
  --json-out /tmp/moss_production_server_opt_artifact_gate.json

# Prepare the canonical /opt layout without copying the large model bundle.
# Default is dry-run; execution requires --execute plus the confirmation string.
python models/tts/moss/prepare_moss_ort_deployment.py \
  --source /home/cat/moss-onnx-baseline \
  --destination /opt/tts/models/moss-tts-nano-onnx \
  --json-out /tmp/moss_ort_deployment_dry_run.json

CONFIG=configs/rk3576-moss-hybrid-rknn-stream.yaml \
python -m rkvoice_stream.app.server

python models/tts/moss/verify_moss_rknn_artifacts.py \
  --model-dir /opt/tts/models/moss-tts-nano-rknn \
  --require-production-default

CONFIG=configs/rk3576-moss-rknn-stream.yaml \
python -m rkvoice_stream.app.server

pytest tests/test_tts.py tests/test_roundtrip.py
```

The production profile verifier is the preferred acceptance command. It validates the ORT artifact manifest, starts the service and checks `/tts/stream` plus `/dialogue`, then runs the isolated TTS -> ASR roundtrip quality gate.

## Current RK3576 Evidence

The Jetson ONNX bundle in WSL2 is reusable as the RKNN conversion source:

```text
wsl2-local:/home/harve/models/moss-onnx-bundle-paged-fp16
```

Validated conversion host:

```text
wsl2-local:/home/harve/rknn-build/.venv/bin/python
onnx 1.16.1
rknn-toolkit2 2.3.0/2.3.x
```

ORT streaming baseline status on RK3576:

- RK3576 environment has `onnxruntime 1.24.4`, `numpy`, `soundfile`, `sentencepiece 0.2.0`, and `python-multipart`.
- `moss_tts_prefill.onnx` + `moss_tts_global_shared.data` run on RK3576 CPU with ORT. Session load is about 2.1-2.3 s; prefill seq32 runtime is about 197-220 ms and outputs are finite.
- `moss_tts_decode_step.onnx` runs with prefill KV on RK3576 CPU with ORT. Decode step runtime is about 57 ms and outputs are finite.
- `moss_tts_local_fixed_sampled_frame.onnx` runs on RK3576 CPU with ORT. Session load is about 5.8-6.0 s; sampler runtime is about 86-92 ms and returns `should_continue=1` plus 16 audio frame token ids.
- `moss_audio_tokenizer_decode_full.onnx` runs on RK3576 CPU with ORT. Session load is about 2.4 s; one-frame decode runtime is about 26-31 ms and returns `audio` shape `[1, 2, 3840]` with RMS about `0.0386`.
- `moss_audio_tokenizer_decode_step.onnx` runs as the production stream codec. Per-frame codec runtime is about 67-69 ms and remains constant across generated frames.
- Formal verifier command used on RK3576:

```bash
/home/cat/rknn-venv/bin/python /tmp/verify_moss_ort_stream.py \
  --model-dir /home/cat/moss-onnx-baseline \
  --json-out /home/cat/moss-onnx-baseline/ort_stream_verify.json
```

Verifier result:

```json
{
  "stream_ms": {
    "prefill": 204.468,
    "sampler": 92.461,
    "codec": 30.985,
    "ttfa": 327.914
  },
  "audio": {
    "shape": [1, 2, 3840],
    "rms": 0.038561
  },
  "gates": {
    "max_ttfa_ms": 500.0,
    "min_rms": 0.02,
    "passed": true
  }
}
```

Service backend smoke after wiring `moss_ort` into `rkvoice_stream.engine.tts`:

```text
preload_ms=12563.3
chunk 1 [3840, 2] ttfa_ms=352 prefill_ms=219.169 sampler_ms=101.560 codec_ms=30.917
chunk 2 [3840, 2] decode_ms=53.005 sampler_ms=79.173 codec_ms=28.575
done chunks=2 frames=7680 wall_ms=540.342
```

Current real text backend smoke with full official prompt, instance-level RNG, streaming codec, 6 ORT threads, and startup warmup:

```text
voice=Junhao, seed=314
preload_ms=15953.8
chunk 1 [3840, 2] ttfa_ms=1081 prefill_ms=886.607 sampler_ms=97.371 codec_ms=92.790
chunk 2 [3840, 2] decode_ms=76.248 sampler_ms=83.414 codec_ms=95.227
done chunks=2 wall_ms=1572.207
```

ASR roundtrip gate on RK3576, using `models/tts/moss/verify_moss_ort_roundtrip.py` and separate TTS/ASR processes to avoid 8GB OOM. The verifier seed is production-style: it initializes the backend RNG once with `MOSS_ORT_SEED` and lets the stream advance across utterances; it does not reset the RNG for every sentence.

```bash
cd /home/cat/rkvoice-stream
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_ort_roundtrip.py \
  --model-dir /home/cat/moss-onnx-baseline \
  --asr-model-dir /home/cat/sherpa-onnx-paraformer-zh-2023-09-14 \
  --out-dir /tmp/moss_ort_junhao_seed_314_confirm \
  --json-out /tmp/moss_ort_junhao_seed_314_confirm.json \
  --threads 6 \
  --voice Junhao \
  --seed 314 \
  --prefill-seq 0 \
  --codec-streaming 1 \
  --manifest moss-ort-manifest.json \
  --warmup-text 你好
```

Verifier result:

```text
你好 -> 你好, CER 0.0
欢迎使用语音服务 -> 欢迎, CER 0.75
语音识别测试一二三四五 -> 语音识别啊, CER 0.636
avg_cer=0.462, max_cer=0.75, min_rms=0.0858
max_ttfa_ms=1070, max_codec_ms=87.401
gates.passed=true
```

This run used `MOSS_ORT_MANIFEST=moss-ort-manifest.json`, so the quality gate is tied to the hash-verified ONNX bundle. It matches the Lingyu ORT baseline quality while reducing max TTFA from about `1743 ms` to about `1070 ms` on the same RK3576 verifier. It still does not satisfy stricter per-sentence production quality because the Paraformer roundtrip recognizes only `欢迎` for the second test sentence; however, it passes the current average-CER fallback gate and is the fastest verified production fallback profile so far. The older 32-token prefill crop reached `ttfa_ms≈363` but failed ASR quality; it is not production safe.

Lingyu production RNG seed sweep on RK3576:

```text
seed=7:    avg_cer=0.583, failed, hypotheses=[你好, 欢迎, 云石地]
seed=42:   avg_cer=0.462, passed, hypotheses=[你好, 欢迎, 语音识别]
seed=99:   avg_cer=0.583, failed, hypotheses=[你好, 欢迎来到, 允许你]
seed=1234: avg_cer=0.462, passed, hypotheses=[你好, 欢迎, 语音识别]
seed=2024: avg_cer=0.583, failed, hypotheses=[你好, 欢迎, 浴室]
seed=2026: avg_cer=0.523, failed, hypotheses=[你好, 欢迎生, 语音鞋]
```

Lingyu `MOSS_ORT_SEED=1234` remains the best long-prompt reference baseline, but it is no longer the default because Junhao seed 314 passes the same gate at much lower TTFA.

ORT threading sweep, original Lingyu profile:

```text
threads=2: max_ttfa_ms≈3049, too slow
threads=3: max_ttfa_ms≈2226, too slow
threads=4: max_ttfa_ms≈2093, previous baseline
threads=5: max_ttfa_ms≈1988
threads=6: max_ttfa_ms≈1753 in TTS sweep; 1710-1783 in full roundtrip, selected
threads=7: max_ttfa_ms≈1679 but sampler/codec jitter increased
threads=8: max_ttfa_ms≈1667 but steady-state RTF and codec jitter worsened
```

ORT threading sweep, current Junhao seed 314 production profile, warmup enabled,
two streamed chunks per run:

```text
threads=4: ttfa=1246/1254 ms, avg=1250 ms
threads=5: ttfa=1116/1160 ms, avg=1138 ms
threads=6: ttfa=1011/1045 ms, avg=1028 ms
threads=7: ttfa=1009/999 ms, avg=1004 ms, codec spike to 122 ms
threads=8: ttfa=1045/945 ms, avg=995 ms, sampler/codec jitter higher
```

Steady 8-frame streaming comparison for the same profile:

```text
threads=6: ttfa=1024 ms, wall=2821.798 ms, decode_max=68.928 ms, sampler_max=91.978 ms, codec_max=106.312 ms
threads=7: ttfa=921 ms, wall=2940.341 ms, decode_max=99.672 ms, sampler_max=103.833 ms, codec_max=105.830 ms
threads=8: ttfa=1098 ms, wall=3410.942 ms, decode_max=98.728 ms, sampler_max=178.603 ms, codec_max=135.895 ms
```

Keep `threads=6`: it is slightly slower than 7 on first payload in one short
run, but it has better total streaming wall time and lower sampler/decode
jitter. For low-latency dialogue, stable chunk cadence matters more than a
single best-case first-payload sample.

Per-session ORT thread sweep, same Junhao seed 314 profile, eight streamed
chunks per run:

```text
base:  prefill=6 decode=6 sampler=6 codec=6 -> ttfa=1043 ms, wall=2875.000 ms, codec_max=91.009 ms
p7:    prefill=7 decode=6 sampler=6 codec=6 -> ttfa=987 ms,  wall=2780.833 ms, codec_max=81.546 ms
p8:    prefill=8 decode=6 sampler=6 codec=6 -> ttfa=950 ms,  wall=2697.395 ms, codec_max=84.384 ms
p8d5:  prefill=8 decode=5 sampler=6 codec=6 -> ttfa=952 ms,  wall=2770.313 ms, codec_max=94.471 ms
p8c5:  prefill=8 decode=6 sampler=6 codec=5 -> ttfa=943 ms,  wall=2704.222 ms, codec_max=84.115 ms
p8s5c5: prefill=8 decode=6 sampler=5 codec=5 -> ttfa=950 ms, wall=2667.520 ms, codec_max=85.048 ms
```

20-frame follow-up smoke for steady-state service length:

```text
p8c5:   prefill=8 decode=6 sampler=6 codec=5 -> ttfa=899 ms, wall=5186.341 ms
p8s5c5: prefill=8 decode=6 sampler=5 codec=5 -> ttfa=968 ms, wall=5212.136 ms
p8s4c5: prefill=8 decode=6 sampler=4 codec=5 -> ttfa=930 ms, wall=5148.690 ms
p8s5c4: prefill=8 decode=6 sampler=5 codec=4 -> ttfa=956 ms, wall=5130.473 ms
p8d5c5: prefill=8 decode=5 sampler=6 codec=5 -> ttfa=918 ms, wall=5098.881 ms
p8d4c5: prefill=8 decode=4 sampler=6 codec=5 -> ttfa=963 ms, wall=5157.569 ms
```

Promote `prefill_threads=8`, `decode_threads=5`, and `codec_threads=5` in the
ORT fallback profile. The 8-frame smoke suggested decode narrowing was only a
small steady-state gain, but the full production verifier below showed better
service first-payload, wall time, roundtrip TTFA, and codec latency with no
quality regression. Leave sampler on the stable base thread count.

Sampler ORT profile on RK3576, using
`models/tts/moss/profile_moss_sampler_ort.py` against
`moss_tts_local_fixed_sampled_frame.onnx`:

```text
threads=6, runs=20: mean=133.938 ms, p50=133.093 ms, p95=142.693 ms, max=145.103 ms
top ops by ORT node time:
MatMul=56.80%, Unsqueeze=3.83%, Add=3.81%, Reshape=3.71%, Slice=2.99%,
Concat=2.81%, Where=2.05%, Gather=1.96%, Shape=1.89%, Transpose=1.88%,
Softmax=0.94%, TopK=0.54%
top nodes:
/text_lm_head/MatMul ~= 4.9 ms/run
audio MLP fc_in/fc_out MatMuls ~= 0.95-1.02 ms/run each, repeated across heads
```

This rules out a pure CPU postprocess rewrite as the main sampler optimization:
TopK/Softmax are under 2% combined, while MatMul dominates. The viable future
split is a narrow RKNN island around the sampler's audio/text linear projections
or MLP path, not moving CDF sampling out of ONNX. A monolithic sampler RKNN is
still not production safe because the current artifact can build/load but fails
true RK3576 inference probes.

Sampler RKNN island probes, extracted from the Jetson/WSL2 ONNX bundle with
`models/tts/moss/extract_moss_rknn_island.py` and verified on RK3576 with
`probe_moss_rknn_runtime.py` plus `verify_moss_rknn_island_parity.py`:

```text
sampler_audio_head0:
  shape: [1, 768] -> [1, 1024], RKNN size=1.6 MB
  runtime probe: OK, infer_ms=2.517, finite output
  parity: rel_l2=0.000291, cosine=1.0, passed
  latency: ORT avg=1.194 ms, RKNN avg=1.389 ms

sampler_audio_heads0-15:
  shape: 16 * [1, 768] -> 16 * [1, 1024], RKNN size=25 MB
  runtime probe: OK, infer_ms=11.821, finite outputs
  parity: max rel_l2≈0.000365, min cosine≈0.99999988, passed
  latency: ORT avg=5.639 ms, RKNN avg=10.068 ms

sampler_fc_out0:
  shape: [1, 1, 3072] -> [1, 1, 768], RKNN size=4.8 MB
  runtime probe: OK, infer_ms=3.376, finite output
  parity: rel_l2=0.000297, cosine=0.99999994, passed
  latency: ORT avg=1.695 ms, RKNN avg=1.472 ms

sampler_fc_in_act0:
  shape: [1, 1, 768] -> [1, 1, 3072], RKNN size=4.8 MB
  runtime probe: OK, infer_ms=3.362, finite output
  parity: rel_l2=0.000546, cosine=0.99999982, passed
  latency: ORT avg=2.646 ms, RKNN avg=2.132 ms

sampler_mlp0:
  shape: [1, 1, 768] -> [1, 1, 768], RKNN size=9.5 MB
  runtime probe: OK, infer_ms=4.295, finite output
  parity: rel_l2=0.000608, cosine=0.99999988, passed
  latency: ORT avg=3.261 ms, RKNN avg=2.273 ms

sampler_mlps0:
  shape: 17 * [1, 1, 768] -> 17 * [1, 1, 768], RKNN size=154 MB
  runtime probe: OK, infer_ms=29.309, finite outputs
  parity: max rel_l2=0.000678, min cosine=0.99999970, passed
  latency: ORT avg=30.279 ms, RKNN avg=21.719 ms

sampler_text_lm_head:
  shape: [1, 768] -> [1, 16384], RKNN size=25 MB
  runtime probe: OK, infer_ms=8.526, finite output
  parity: rel_l2=0.000296, cosine=0.99999934, passed
  latency: ORT avg=4.900 ms, RKNN avg=2.948 ms
```

Conclusion: sampler head RKNN is accurate but slower than ORT, even when all 16
audio heads are grouped into one RKNN call. The useful sampler RKNN candidates
are now the fused local MLPs and text LM head: both are accurate and
individually faster than ORT. This still does not justify service integration by
itself. `sampler_mlps0` saves about `8.6 ms` per sampler frame and
`sampler_text_lm_head` saves about `2 ms`, while a production integration must
preserve the local transformer's sequential hidden-state dependencies, exact
sampler decisions, and avoid extra CPU/NPU handoff jitter. RK3576 root disk was
down to a few hundred MB after these probes, so larger all-layer sampler
artifacts should be built on WSL2 and transferred only after space is cleared or
a non-root storage path is available.

Disk audit note: `/tmp` is a separate `tmpfs` on the current RK3576 image. The
MOSS `/tmp` cleanup flow can remove stale temporary probe directories, but it
does not increase `/` free space. `audit_moss_disk.py` reports
`tmp_candidates_affect_root_disk=false` and `home_candidates_affect_root_disk=true`
on the device; root-disk recovery must target reviewed `/home/cat` candidates or
move future RKNN artifacts to external/non-root storage.
The classified disk report protects known runtime dependencies such as
`/home/cat/moss-onnx-baseline`, `/home/cat/rknn-venv`, and
`/home/cat/sherpa-onnx-paraformer-zh-2023-09-14`. On the latest RK3576 audit,
`home_review_archive_or_move_mb` was about `9732 MB`; these are experiment,
export, probe, `.onnx`, or `.rknn` artifacts that should be reviewed, archived,
or moved off root storage before more RKNN probe runs.
The report also emits `home_migration_plan`, with `migration_priority` and
`suggested_action` for each reviewable item. No `/home` item is marked safe to
delete without review. Current high-priority migration candidates include
`/home/cat/moss-rknn-probe`, `/home/cat/moss-official-bundle`,
`/home/cat/paraformer-encoder.onnx`, and `/home/cat/qwen3-tts-export`.
Storage audit also found an unmounted `mmcblk1` device of about `115 GB` with
no `blkid` filesystem entry. This is the best candidate for future RKNN artifact
workspace, but using it requires explicit approval to partition/format/mount the
device; do not write RKNN experiments back to `/home/cat` until either that
storage is prepared or enough reviewed `/home/cat` artifacts have been moved
off root storage. `audit_moss_disk.py` now reports this as
`unmounted_block_device_candidates`, including `requires_destructive_setup=true`
for `/dev/mmcblk1`.
The preparation flow is scripted but defaults to dry-run:

```bash
python models/tts/moss/prepare_rknn_workspace.py \
  --device /dev/mmcblk1 \
  --mount-point /mnt/rknn-workspace \
  --json-out /tmp/rknn_workspace_dry_run.json
```

Latest RK3576 dry-run evidence is saved at
`docs/evidence/moss/rk3576-rknn-workspace-dry-run.json`: `passed=true`,
`execute=false`, `missing_commands=[]`, `persist_fstab=true`, and every planned
command has `executed=false`. The dry-run plan resolves the target partition as
`/dev/mmcblk1p1` and now includes the persistent fstab entry
`LABEL=RKNN_WS /mnt/rknn-workspace ext4 defaults,nofail,noatime 0 2`, so an
approved workspace setup will survive reboot instead of creating only a
temporary mount. This proves the board has the required preparation tools
installed without having partitioned or formatted `/dev/mmcblk1`.
The preparation script also handles partition naming for non-mmc device families
(`sda -> sda1`, `nvme0n1 -> nvme0n1p1`) and verifies the target is an existing
block device before executing destructive commands.
Full RKNN conversion must also prove it is writing under the prepared workspace:

```bash
python models/tts/moss/convert_moss_rknn.py \
  --onnx-bundle /path/to/moss-onnx-bundle \
  --out-dir /mnt/rknn-workspace/moss-rknn-workspace/moss-tts-nano-rknn \
  --require-rknn-workspace \
  --rknn-workspace /mnt/rknn-workspace/moss-rknn-workspace
```

With `--require-rknn-workspace`, conversion refuses to start unless the workspace
passes `verify_rknn_artifact_workspace.py` and `--out-dir` is under that
workspace. Do not run new large conversions with `--out-dir /home/cat/...`.
Future RKNN conversion/probe runs should pass
`models/tts/moss/verify_rknn_artifact_workspace.py` before writing large
artifacts. Current RK3576 evidence shows that root space has recovered to about
`15.6 GB`, but `/home/cat` and `/mnt` are still on the root filesystem and are
therefore rejected for new large RKNN artifact workspaces:

`models/tts/moss/verify_rknn_workspace_deployment.py` is the post-setup
deployment verifier. It checks the persistent fstab entry, the actual mount,
and the writable non-root workspace in one report. Current RK3576 evidence
correctly fails because the destructive setup has not been approved/executed:

```text
python models/tts/moss/verify_rknn_workspace_deployment.py
passed=false
fstab_entry_present=false
mounted=null
errors=[
  "missing fstab entry: LABEL=RKNN_WS /mnt/rknn-workspace ext4 defaults,nofail,noatime 0 2",
  "mount point is not mounted: /mnt/rknn-workspace",
  "workspace does not exist: /mnt/rknn-workspace/moss-rknn-workspace",
  "workspace is on the root filesystem: /mnt/rknn-workspace/moss-rknn-workspace"
]
repo_json=docs/evidence/moss/rk3576-rknn-workspace-deployment-not-ready.json
```

```text
python models/tts/moss/verify_rknn_artifact_workspace.py \
  --workspace /home/cat/moss-rknn-probe \
  --min-free-mb 2048
passed=false
errors=[
  "workspace does not exist: /home/cat/moss-rknn-probe",
  "workspace is under home directory; move RKNN artifacts off root storage: /home/cat/moss-rknn-probe",
  "workspace is on the root filesystem: /home/cat/moss-rknn-probe"
]
free_mb=15585
repo_json=docs/evidence/moss/rk3576-rknn-workspace-home-rejected.json

python models/tts/moss/verify_rknn_artifact_workspace.py \
  --workspace /mnt/rknn-workspace/moss-rknn-workspace \
  --min-free-mb 2048
passed=false
errors=[
  "workspace does not exist: /mnt/rknn-workspace/moss-rknn-workspace",
  "workspace is on the root filesystem: /mnt/rknn-workspace/moss-rknn-workspace"
]
free_mb=15585
repo_json=docs/evidence/moss/rk3576-rknn-workspace-mnt-not-ready.json
```

This makes the storage constraint explicit: do not generate or transfer new
large RKNN artifacts to `/home/cat` just because there is temporary free space.
Prepare `/dev/mmcblk1` or another non-root workspace before the next large
conversion/probe cycle.
The same workspace requirement can be promoted into the release/preflight audit
when doing RKNN acceleration work:

```text
python models/tts/moss/audit_moss_release.py --require-rknn-workspace
passed=false
ort_config=true
ort_evidence=true
production_entrypoint=true
rknn_candidate_config=true
rknn_workspace=false
error=workspace does not exist: /mnt/rknn-workspace/moss-rknn-workspace
error=workspace is on the root filesystem: /mnt/rknn-workspace/moss-rknn-workspace
repo_json=docs/evidence/moss/rk3576-moss-release-audit-rknn-workspace-required.json

python models/tts/moss/run_moss_production_server.py --dry-run --require-rknn-workspace
passed=false
rknn_workspace=false
repo_json=docs/evidence/moss/rk3576-moss-production-server-rknn-workspace-required.json

python models/tts/moss/audit_moss_release.py --require-rknn-workspace-deployment
passed=false
ort_config=true
production_entrypoint=true
rknn_candidate_config=true
rknn_workspace_deployment=false
fstab_entry_present=false
mounted=null
repo_json=docs/evidence/moss/rk3576-moss-release-audit-rknn-workspace-deployment-required.json

python models/tts/moss/run_moss_production_server.py --dry-run --require-rknn-workspace-deployment
passed=false
rknn_workspace_deployment=false
fstab_entry_present=false
mounted=null
repo_json=docs/evidence/moss/rk3576-moss-production-server-rknn-workspace-deployment-required.json
```
The current RK3576 release audit with `--min-root-free-mb 512` is saved at
`docs/evidence/moss/rk3576-moss-release-audit-current.json`; all MOSS config,
streaming, roundtrip, audited production-entrypoint, and root-disk gates pass.
This removes the immediate production-start blocker, but it does not make root
storage an approved workspace for large RKNN artifacts.
The audited production service runner dry-run is saved at
`docs/evidence/moss/rk3576-moss-production-server-dry-run.json`; it passes the
release gate and resolves the exact uvicorn command with
`CONFIG=configs/rk3576-moss-ort-stream.yaml`. The same runner with the root-disk
gate enabled is saved at
`docs/evidence/moss/rk3576-moss-production-server-disk-gate.json` and currently
passes with `free_mb=15585`.

Execution requires explicit approval because it partitions and formats the
device:

```bash
python models/tts/moss/prepare_rknn_workspace.py \
  --device /dev/mmcblk1 \
  --mount-point /mnt/rknn-workspace \
  --execute \
  --confirm FORMAT_RKNN_WORKSPACE \
  --json-out /tmp/rknn_workspace_prepare.json
```

`models/tts/moss/verify_moss_sampler_text_head_split.py` validates the first
full-sampler replacement attempt: full ORT vs `prefix ORT -> text_lm_head RKNN
-> suffix ORT`. The text head RKNN output remained numerically close and final
sampler decisions matched:

```text
runs=4
token_equal=4/4
continue_equal=4/4
max text-logit rel_l2=0.000292
min text-logit cosine=0.99999928
```

However, this is not a production-speed split yet:

```text
full_ort_avg=86.522 ms
prefix_ort_avg=84.217 ms
rknn_head_avg=3.204 ms
suffix_ort_avg=90.504 ms
split_total_avg=177.925 ms
```

The verifier intentionally rewrites the ONNX graph without copying external
data, but the prefix/suffix graphs are not truly pruned; ORT still does nearly a
full sampler pass on each side. This proves the numerical split boundary is
safe, but it also proves that backend integration must use extracted/pruned
subgraphs or an in-process sampler implementation. Do not wire this naive split
into the service.

The same verifier can now prepare true pruned subgraphs on WSL2 with
`--prepare-only --split-mode pruned`; the generated prefix and suffix are then
transferred into the RK3576 model directory so ORT external-data validation
passes without copying `moss_tts_local_shared.data`. Pruned split result on
RK3576:

```text
runs=8
token_equal=8/8
continue_equal=8/8
max text-logit rel_l2=0.000292
min text-logit cosine=0.99999928

full_ort_avg=82.590 ms
prefix_ort_avg=7.664 ms
rknn_head_avg=8.179 ms
suffix_ort_avg=82.696 ms
split_total_avg=98.539 ms
```

This is numerically safe and much better than the naive split, but still slower
than full ORT because the pruned suffix contains almost the whole audio-token
sampling tail. Do not promote a text-head-only split. A production sampler
acceleration must either move more of the suffix into RKNN, especially the fused
local MLP and/or audio-token logits path, or replace the suffix with an
equivalent CPU implementation that avoids a second ORT graph dispatch.

Pruned text-head suffix profile on RK3576 confirms where the remaining suffix
time goes:

```text
threads=6, runs=12
suffix mean=104.629 ms, p50=104.200 ms, p95=106.087 ms, max=107.883 ms
graph inputs: global_hidden, repetition_seen_mask, assistant_random_u,
  audio_random_u, /text_lm_head/MatMul_output_0
graph outputs: should_continue, frame_token_ids
top ops:
Gemm=54.14%, MatMul=7.18%, Reshape=6.35%, Slice=2.86%,
Unsqueeze=2.69%, Where=2.68%, Softmax=1.20%, TopK=0.70%
top nodes are /mlp/fc_in_* and /mlp/fc_out_* MatMulAddFusion,
about 0.95-0.99 ms each per profile event.
```

This makes the next target concrete: the suffix bottleneck is the 17 local MLP
blocks, not TopK/CDF sampling. A pure Python or NumPy rewrite of postprocess
alone will not move the service-level number enough.

`models/tts/moss/verify_moss_sampler_text_head_mlps_split.py` tests a wider
split:

```text
full ORT sampler
vs prefix ORT -> text_head RKNN + 17xMLP RKNN -> suffix ORT
```

RK3576 result with `sampler_text_lm_head` and `sampler_mlps0`:

```text
runs=4
token_equal=2/4
continue_equal=4/4
MLP parity: max rel_l2=0.003201, min cosine=0.99999493, passed

full_ort_avg=87.091 ms
prefix_ort_avg=75.764 ms
rknn_text_avg=6.600 ms
rknn_mlps_avg=25.628 ms
suffix_ort_avg=39.281 ms
split_total_avg=147.273 ms
gates.passed=false
```

Do not promote this split. The failure is structural, not just numerical:
dumping all 17 `ln_2` tensors from one ORT prefix and then replacing all 17 MLP
outputs does not preserve the local transformer's sequential dependency chain.
Later `ln_2_i` tensors in that prefix have already been computed through the ORT
MLP path. A valid sampler split must execute per local block:
ORT block prefix to `ln_2_i` -> RKNN MLP_i -> residual/update hidden -> next
block. Only after that sequential split passes token parity should it be wired
into `moss_ort`.

`models/tts/moss/verify_moss_sampler_sequential_mlps_split.py` implements that
dependency-preserving control. It creates temporary stage ONNX files in the
model directory so external-data validation sees a regular
`moss_tts_local_shared.data` file; the temporary directory is deleted after the
run. The verifier intentionally measures correctness first, not production
speed, because it currently loads many ORT stage sessions and calls the grouped
17-MLP RKNN once per local block.

Sequential split result with `sampler_text_lm_head` and `sampler_mlps0` on
RK3576:

```text
runs=2
token_equal=1/2
continue_equal=2/2
MLP parity: max rel_l2=0.003201, min cosine=0.99999493, passed

full_ort_avg=107.341 ms
stage_ort_avg=1196.564 ms
rknn_text_avg=3.500 ms
rknn_mlps_avg=394.055 ms
suffix_ort_avg=44.731 ms
split_total_avg=1638.850 ms
gates.passed=false
```

Sequential ORT-MLP control, same stage graph and text-head RKNN, but with MLP
outputs computed by the extracted ORT `sampler_mlps0.onnx`:

```text
runs=2
token_equal=2/2
continue_equal=2/2
full_ort_avg=86.381 ms
stage_ort_avg=1191.662 ms
rknn_text_avg=4.693 ms
ort_mlps_avg=724.527 ms
suffix_ort_avg=43.303 ms
split_total_avg=1964.186 ms
gates.passed=true
```

This closes the sampler split diagnosis: the sequential graph surgery is
correct, and text-head RKNN is not the observed parity blocker. RKNN FP16 local
MLP drift is enough to change stochastic audio-code sampling even when cosine
parity is high. The grouped `sampler_mlps0` artifact remains useful as a
performance/accuracy probe, but it is not production safe for MOSS sampler
replacement on RK3576 without a correction strategy or a numerically tighter
runtime path. Also, using the grouped 17-output RKNN once per local block is far
too slow.

`verify_moss_sampler_sequential_mlps_split.py` now also supports the next
production-oriented sampler experiment: pass `--mlps-rknn-dir <dir>` containing
17 `moss_sampler_mlp<N>*.rknn` per-block artifacts instead of the grouped
`--mlps-rknn` artifact. That keeps the same dependency-preserving verifier while
removing the zero-filled 17-input grouped-runner overhead. This is the next
sampler path to test before considering any service integration.

Per-block sampler MLP RKNN result on RK3576:

```text
artifacts: 17 x moss_sampler_mlp<N>.fp16.rk3576.rknn, 9.1 MB each, 155 MB total on device
runs=1
token_equal=0/1
continue_equal=1/1
token_mismatches=13/16
MLP parity: max rel_l2=0.003140, min cosine=0.99999505, passed

full_ort_avg=92.728 ms
stage_ort_avg=1228.124 ms
rknn_text_avg=3.582 ms
rknn_mlps_avg=72.729 ms
suffix_ort_avg=38.494 ms
split_total_avg=1342.929 ms
gates.passed=false
json=/tmp/moss_sampler_sequential_per_block_rknn_runs1.json
```

This confirms the per-block runner fixes the gross performance issue of the
grouped 17-output RKNN (`rknn_mlps_avg≈394 ms -> 72.729 ms`), but it does not
fix sampler correctness. The remaining blocker is numerical sensitivity: RKNN
FP16 MLP output drift is small by cosine/rel-L2, yet still changes stochastic
audio-code sampling. Do not wire this sampler split into `moss_ort` until a
token-level correction, tighter precision path, or sampling strategy preserves
token parity and then passes ASR roundtrip quality.

The sequential verifier now emits an explicit `promotion` decision. A rerun
against the canonical `/opt` model path and the existing per-block RKNN MLP
artifacts still blocks service integration:

```text
runs=1
token_equal=0/1
continue_equal=1/1
gates: token_parity=false continue_parity=true mlp_parity=true passed=false
latency: full_ort_avg=76.474 ms split_total_avg=1328.762 ms rknn_mlps_avg=66.613 ms
promotion.allow_service_integration=false
promotion.errors=[
  "token parity failed: 0/1",
  "split speedup 0.058x below required 1.050x"
]
json=docs/evidence/moss/rk3576-moss-sampler-sequential-per-block-rknn-promotion.json
```

This keeps the RKNN sampler experiment from being mistaken for a service-ready
optimization: even though the per-block RKNN MLP runner is much faster than the
old grouped runner, it is still not exact and the current verifier architecture
is far slower than full ORT. The next valid sampler path is a numerically tighter
MLP/logit strategy or a different split that proves exact token parity and a
real speedup before any backend wiring.

Layer attribution using the same dependency-preserving verifier:

```text
all RKNN MLP layers:
  token_equal=0/1, mismatches=13/16, mismatch_indices=3..15
  rknn_mlps_avg=64.622 ms

single/range probes, one run each:
  layers 0-3:   passed, mismatches=0
  layer 4:      failed, mismatches=13, mismatch_indices=3..15
  layer 5:      passed, mismatches=0
  layer 6:      passed, mismatches=0
  layer 7:      passed, mismatches=0
  layer 8:      passed, mismatches=0
  layer 9:      passed, mismatches=0
  layer 10:     failed, mismatches=6, mismatch_indices=9,10,11,13,14,15
  layer 11:     failed, mismatches=6, mismatch_indices=10..15
  layers 12-16: passed, mismatches=0
```

The sensitive local blocks for this seed are `4`, `10`, and `11`, but excluding
only those is not enough for production. The apparent safe subset
`0-3,5-9,12-16` passed the first run and then failed the 4-run gate
(`token_equal=1/4`, later runs had 14, 13, and 3 token mismatches). A more
conservative `12-16` subset also failed the 4-run gate (`token_equal=3/4`).
Therefore sampler MLP RKNN is not production-safe even as a partial layer
replacement under the current FP16 runtime. The next sampler direction should be
margin-aware correction or deterministic sampling constraints, not just choosing
a smaller static set of RKNN MLP layers.

Margin debug for the `layer 4` failure shows this is not merely a random draw
falling close to a CDF boundary:

```text
layers=4
token_equal=0/1
token_mismatches=13/16
mismatched_full_random_margin_min=0.000412
mismatched_full_random_margins=[
  0.016053, 0.006381, 0.010360, 0.001200, 0.003919,
  0.005635, 0.022985, 0.012771, 0.013770, 0.003963,
  0.008440, 0.004736, 0.000412
]
mismatched_full_token_missing_from_hybrid_topk=8/13
mismatched_hybrid_token_missing_from_full_topk=6/13
json=/tmp/moss_sampler_layer4_margin_debug.json
```

Some mismatches do happen near CDF boundaries, but most are stronger distribution
changes: the ORT-selected token is often absent from the RKNN split TopK. A
simple "fallback only when random margin is tiny" rule would miss these cases.
Any correction path must either preserve the sampler logits/TopK more directly
or include a broader confidence signal than CDF-boundary margin.

Sampler-only thread sweep on the same profiler:

```text
threads=4: mean=143.885 ms, p50=144.436 ms, p95=146.082 ms, max=146.289 ms
threads=5: mean=138.629 ms, p50=139.520 ms, p95=140.707 ms, max=147.370 ms
threads=6: mean=140.115 ms, p50=137.497 ms, p95=140.813 ms, max=175.946 ms
threads=7: mean=132.976 ms, p50=130.801 ms, p95=134.518 ms, max=167.009 ms
threads=8: mean=137.993 ms, p50=134.110 ms, p95=136.964 ms, max=183.386 ms
```

`sampler_threads=7` passed the full production verifier but was not promoted:

```text
/tts/stream first_payload=936.568 ms, wall=4535.587 ms
/dialogue first_payload=996.448 ms, wall=4608.630 ms, binary_chunks=7
roundtrip avg_cer=0.462121, max_cer=0.75, min_rms=0.085809
roundtrip max_ttfa=1003 ms, max_codec=150.270 ms
gates.passed=true
```

Compared with the current `sampler_threads=6` production evidence, this is mixed:
TTS first payload is slightly better, but dialogue wall and roundtrip TTFA are
worse. Keep sampler on the base `threads=6` until a full verifier shows a clear
service-level win.

Streaming codec batch probe on RK3576:

```text
frames=8 batch_frames=4
single codec calls=8 total_ms=537.879 max_ms=86.646
batch codec calls=2 total_ms=286.636 max_ms=143.609
max_abs_diff=1.214e-6
passed=true
```

`codec_batch_frames=3` is now enabled in the ORT fallback profile. The first
audio frame still flushes immediately; later codec calls batch up to three token
frames. This keeps first-audio latency low while reducing codec call count and
steady-state wall time without the four-frame jitter seen in the backend-stage
gate. Four-frame batching remains useful as a throughput probe, but the latest
production gate failed on `max_codec_ms=210.116`; three-frame batching passed the
same backend-stage gate with `max_codec_ms=130.455`.

Do not enable asynchronous codec overlap on RK3576 by default. A prototype
showed the audio can remain bit-equivalent, but running codec in a worker while
decode/sampler continue causes CPU contention on RK3576: four-frame codec calls
rose from about `140-153 ms` to `380-466 ms`, and the 20-frame smoke wall time
regressed to `5001.351 ms`. Keep codec execution synchronous unless a future
runtime or pinning strategy proves a production verifier improvement.

CPU affinity was tested and should not be enabled by default. RK3576 maps CPUs `0-3` to Cortex-A53 and `4-7` to Cortex-A72. Pinning to A72 only (`taskset -c 4-7`) made `threads=4` slower (`max_ttfa_ms≈2112`), and running `threads=6` on `4-7` badly oversubscribed the big cores (`max_ttfa_ms≈4389`). Pinning to `2-7` also worsened steady-state codec/sampler latency. Let Linux schedule across all cores and use `threads=6`.

Shorter built-in voices reduce TTFA but can fail content quality. Initial two-seed sweeps for several voices showed Junhao was the most promising:

```text
Junhao seed 42:   max_ttfa_ms=1098, avg_cer=0.523, failed
Yuewen seed 42:   max_ttfa_ms=1115, avg_cer=0.523, failed
Weiguo seed 1234: max_ttfa_ms=1320, avg_cer=0.523, failed
Xiaoyu seed 1234: max_ttfa_ms=1532, avg_cer=0.583, failed
Lingyu seed 1234: max_ttfa_ms=1743, avg_cer=0.462, passed
```

Junhao extended seed sweep found several passing seeds:

```text
Junhao seed 314: max_ttfa_ms=1073, avg_cer=0.462, passed
Junhao seed 13:  max_ttfa_ms=1095, avg_cer=0.462, passed
Junhao seed 5:   max_ttfa_ms=1096, avg_cer=0.462, passed
Junhao seed 77:  max_ttfa_ms=1102, avg_cer=0.462, passed
Junhao seed 1:   max_ttfa_ms=1140, avg_cer=0.462, passed
```

Use `voice=Junhao` and `MOSS_ORT_SEED=314` as the RK3576 ORT production fallback. It is the fastest verified combination under the current three-sentence ASR roundtrip gate.

`MOSS_ORT_WARMUP_TEXT=你好` is enabled in the RK3576 service profile. It increases preload time but removes avoidable first-request cold work. Preload is outside the dialogue TTFA budget and should run during service startup.

Junhao prompt tail-crop sweep confirms that `prefill_seq=0` must stay in the
production profile. Junhao full prompt for `你好` is 177 rows (`168` voice rows
+ `9` text/assistant/audio-start rows). Tail-cropping improves TTFA but destroys
semantic quality:

```text
prefill_seq=64:  max_ttfa=497 ms, avg_cer=1.708, failed, hypotheses=[嗯, 二零一零年零一零一一一一一一一一一一一一的一的东个, 真的]
prefill_seq=96:  max_ttfa=665 ms, avg_cer=1.000, failed, hypotheses=[嗯, 嗯嗯, 嗯]
prefill_seq=128: max_ttfa=799 ms, avg_cer=1.000, failed, hypotheses=[对, 看好, 嗯嗯是的是的]
prefill_seq=160: max_ttfa=944 ms, avg_cer=1.000, failed, hypotheses=[嗯, 不爱听见, ""]
prefill_seq=0:   max_ttfa≈1070 ms, avg_cer=0.462, passed, uses full prompt
```

Do not promote cropped prompt profiles for low-latency dialogue unless a future
exporter or prompt format proves quality parity. Even `prefill_seq=160`, which
removes only 17 rows for this prompt, failed the current ASR gate.

`MOSS_ORT_CACHE_VOICE_PREFIX=1` is disabled in the backend. The MOSS prefill output KV length is not equal to the prompt row count for reference-audio prefixes (`Lingyu` example: full rows `300`, cache split rows `232`, prefix KV length `164`), and prefix-KV + `decode_step` hidden states diverged from full prefill (`hidden_rel_l2≈1.0`). Ordinary causal prefix caching is therefore not production safe for this ONNX export; a faster path must come from a verified exporter change, a correct graph split, or RKNN acceleration.

RKNN island status for MOSS prefill:

- Full monolithic MOSS RKNN is not production safe on RK3576. Prefill crops, decode_step, sampler, and codec decode_step can load/init but either crash at `inference()` or fail conversion/runtime in current RKNN 2.3.x paths.
- The stable RKNN path found so far is the transformer block `ln2 + MLP` island, using fixed `seq_len=320` float hidden input and no float `inputs_pass_through`.
- All 12 `ln2_mlp.s320` islands pass RK3576 `rknnlite` inference and parity gates. Random-input aggregate: `sum_ort_avg=740.995 ms`, `sum_rknn_avg=187.194 ms`, `mean_rel_l2=0.002291`, `min_cosine=0.9999961`.
- Real Lingyu + `你好` prefill hidden verification also passes. Actual prefill length is `297` rows padded to bucket `320`; full-prefill ORT tensors vs RKNN island outputs: `mean_rel_l2=0.002197`, `min_cosine=0.9999940`, gate `passed=true`. The 12 MLP islands measure `sum_ort_avg=863.405 ms` vs `sum_rknn_avg=254.599 ms`, for an estimated MLP-only saving of `608.806 ms`.
- Co-resident 12 RKNN contexts fit RK3576 memory: RSS grows from about `31 MB` before load to about `339 MB` after load/init, then about `388 MB` after one sequential 12-layer pass. Sequential steady-state 12-layer runtime is about `203 ms`.

Hybrid split evidence:

- `models/tts/moss/extract_moss_rknn_island.py --preset attn_residual` can extract a CPU ONNX slice for each transformer block's attention + residual path. The slice input is the masked hidden state plus `attention_mask`; outputs are the `ln2_mlp` input and the layer KV tensors.
- `--preset embedding_prefix` extracts `input_ids -> /Add_15_output_0`; `--preset final_norm` extracts `/Mul_88_output_0 -> /ln_f/LayerNormalization_output_0`, with the final output mask applied in Python from `attention_mask`.
- Block0 `attn_residual.s320` was extracted from the Jetson ONNX bundle and transferred to RK3576. ORT loads it with inputs `['/Add_15_output_0[1, 320, 768]', 'attention_mask[1, 320]']` and outputs `['/Add_19_output_0', 'present_key_0', 'present_value_0']`.
- RK3576 real Lingyu + `你好` parity for block0 attention slice passes against the full prefill graph. Actual prefill length is `297`; `/Add_19_output_0` rel_l2 is `7.3e-8`, `present_key_0` rel_l2 is `4.6e-7`, and `present_value_0` rel_l2 is `1.9e-6`. The slice measured `83.039 ms` on RK3576 CPU with 6 ORT threads.
- All 12 `attn_residual.s320` slices now pass RK3576 real Lingyu + `你好` parity against full prefill. Summary artifact: `/home/cat/moss-rknn-probe/attn_slice_lingyu_nihao_s320_summary.json`.
  - `files=12`, `all_passed=true`
  - `sum_slice_ms=887.698`, `avg_slice_ms=73.975`
  - `max_rel_l2=2.98e-6`, `min_cosine=0.99999988`
  - per-layer slice latency range: `65.119-84.344 ms`
- Edge slice parity also passes on RK3576 for real Lingyu + `你好`. `embedding_prefix` is exact (`rel_l2=0`) and measured `43.285 ms`; `final_norm + Python mask` is exact (`rel_l2=0`) and measured `3.139 ms`.
- End-to-end hybrid prefill composition now passes against full ONNX prefill. Verifier artifact: `/home/cat/moss-rknn-probe/hybrid_prefill_lingyu_nihao_s320.json`.
  - composition: `embedding_prefix ORT -> 12 * (attn_residual ORT + ln2_mlp RKNN + CPU residual/mask) -> final_norm ORT + CPU mask`
  - `actual_len=297`, `seq_len=320`
  - full prefill target run: `1643.941 ms`
  - hybrid prefill run after preload: `1115.641 ms`
  - measured saving for full-prompt prefill: `528.300 ms`
  - `global_hidden rel_l2=0.002594`, `global_hidden cosine=0.9999962`
  - KV outputs finite; `kv_max_rel_l2=0.003296`, `kv_min_cosine=0.9999906`
  - hybrid preload for the verifier's ORT+RKNN sessions: `3196.576 ms`

This numerical evidence has been integrated into `moss_ort` as a hybrid prefill path. The backend preloads the verified ORT/RKNN slice sessions, runs hybrid prefill only for the supported fixed bucket, then reuses the existing streaming sampler and `moss_audio_tokenizer_decode_step.onnx` so 80 ms PCM chunks still stream through `synthesize_stream()`. `MOSS_ORT_HYBRID_STRICT=1` is required for verification; without strict mode, the backend may fall back to full ORT if hybrid initialization fails.

Backend-level hybrid smoke on RK3576, using Lingyu + `你好`, strict hybrid, 6 ORT threads, streaming codec, and `max_new_frames=2`:

```text
preload_ms=18660.486
chunk 1 [3840, 2] mode=text_hybrid_rknn ttfa_ms=1370 prefill_ms=1156.278 hybrid_prefill_ms=1151.245 sampler_ms=106.860 codec_ms=94.530
chunk 2 [3840, 2] decode_ms=97.859 sampler_ms=91.666 codec_ms=119.083
done chunks=2 wall_ms=2066.413
```

Compared with the full ORT real-text path (`ttfa_ms≈1658-1702`), the integrated hybrid prefill path cuts first-audio latency by roughly `300 ms` in the current service process. The win is smaller than the isolated prefill saving because sampler, codec, decode-step setup, Python glue, and RKNN/ORT transfers still sit on the first-chunk path.

Hybrid service smoke on RK3576 with the same backend profile:

```text
GET /health -> {"tts":true,"tts_backend":"moss_ort","streaming_tts":true,"asr":false,...}
POST /tts/stream -> 30724 bytes
stream header sample_rate=48000
payload_bytes=30720
int16_samples=15360
```

Hybrid dialogue WebSocket smoke on RK3576:

```text
WS /dialogue {"text":"你好"} -> sample_rate=48000, binary_chunks=3, payload_bytes=30720, done=true
```

The three binary messages are the sample-rate header plus two 80 ms stereo PCM chunks. This validates that the hybrid backend preserves dialogue-mode streaming and does not regress into full-WAV buffering.

Hybrid ASR roundtrip quality gate on RK3576 currently fails and blocks production promotion. The latest run includes the KV trim fix described below:

```bash
cd /home/cat/rkvoice-stream
env PYTHONPATH=/home/cat/rkvoice-stream \
  MOSS_ORT_HYBRID_RKNN=1 \
  MOSS_ORT_HYBRID_STRICT=1 \
  MOSS_ORT_HYBRID_DIR=/home/cat/moss-rknn-probe \
  MOSS_ORT_HYBRID_SEQ_LEN=320 \
  /home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_ort_roundtrip.py \
    --model-dir /home/cat/moss-onnx-baseline \
    --asr-model-dir /home/cat/sherpa-onnx-paraformer-zh-2023-09-14 \
    --out-dir /tmp/moss_hybrid_roundtrip_verify_trim \
    --json-out /tmp/moss_hybrid_roundtrip_verify_trim.json \
    --threads 6 \
    --voice Lingyu \
    --prefill-seq 0 \
    --codec-streaming 1 \
    --warmup-text 你好
```

Verifier result:

```text
你好 -> 你好, CER 0.0
欢迎使用语音服务 -> 欢迎, CER 0.75
语音识别测试一二三四五 -> 林, CER 1.0
avg_cer=0.583, max_cer=1.0, min_rms=0.0500
max_ttfa_ms=1160, max_codec_ms=73.928
gates.passed=false
```

This is a production blocker. The KV trim fix improved quality substantially (`avg_cer 0.917 -> 0.583`) and reduced some over-generation, but it is still worse than the full ORT production fallback (`avg_cer≈0.462`). The remaining failure mode is sampler sensitivity: RKNN FP16 MLP islands have good hidden/KV cosine parity, but small prefill-logit differences can change stochastic local token sampling and degrade semantic content. Keep hybrid behind explicit opt-in flags until token-level or audio-level quality passes the same roundtrip gate as full ORT.

Production MOSS profiles set `tts.require_backend=1`, which maps to
`REQUIRE_TTS_BACKEND=1` at startup. If `moss_ort` or the future `moss_rknn`
backend cannot preload, the service must fail fast instead of starting with TTS
silently disabled. This matters for supervised deployments and RKLLM
co-residency experiments: RKLLM may be added as an optional local dialogue LLM,
but it must not mask or replace a failed MOSS TTS runtime.

Sampler-boundary verifier:

```bash
PYTHONPATH=/home/cat/rkvoice-stream \
/home/cat/rknn-venv/bin/python models/tts/moss/compare_moss_hybrid_sampler.py \
  --model-dir /home/cat/moss-onnx-baseline \
  --artifact-dir /home/cat/moss-rknn-probe \
  --text 你好 \
  --voice Lingyu \
  --seq-len 320 \
  --threads 6 \
  --frames 4 \
  --seed 1234 \
  --rknn-layers all \
  --json-out /tmp/moss_hybrid_sampler_compare_all_trim.json

PYTHONPATH=/home/cat/rkvoice-stream \
/home/cat/rknn-venv/bin/python models/tts/moss/compare_moss_hybrid_sampler.py \
  --model-dir /home/cat/moss-onnx-baseline \
  --artifact-dir /home/cat/moss-rknn-probe \
  --text 你好 \
  --voice Lingyu \
  --seq-len 320 \
  --threads 6 \
  --frames 1 \
  --seed 1234 \
  --rknn-layers all \
  --sampler-debug \
  --json-out /tmp/moss_hybrid_sampler_margin_all_frame1.json
```

Key findings:

- A real backend bug was found and fixed: hybrid prefill returned KV caches padded to bucket length `320` while decode used `past_valid_lengths=297`. Pure ORT slice composition (`--rknn-layers none`) matched full ORT on the first frame but diverged after decode until KV caches were trimmed back to the actual prompt length. After the trim fix, the pure ORT slice control matches full ORT for all 4 tested frames: token mismatches `[0, 0, 0, 0]`.
- With all 12 RKNN MLP layers enabled after the KV trim fix, the first sampler input still has `rel_l2=0.002347` and `cosine=0.9999973`, but first-frame audio-code mismatches are already `9/16`. Later frames remain divergent: token mismatches `[9, 15, 16, 16]`.
- Sampler-margin debug on RK3576 confirms the mismatch is not only a near-boundary random draw. For all 12 RKNN MLP layers on frame 0, `9/16` channels mismatch; among those mismatches, `4` full-ORT tokens are missing from the hybrid TopK candidate set and `5` hybrid tokens are missing from the full-ORT TopK candidate set. The smallest full-ORT random-boundary margin among mismatched channels is about `0.0051`, while the full-frame minimum is about `0.0025`. Control run with `--rknn-layers none --sampler-debug` passes with `0` mismatches, `rel_l2=1.15e-6`, and the same full-frame margin floor. This means the RKNN MLP drift is changing the sampler candidate distribution itself, not merely exposing an unstable random threshold.
- Single-layer RKNN scan does not produce a useful production subset. Only layer 9 matched full ORT for this one text/seed over 4 frames; every other single layer caused sampler token divergence. The composed split path with mostly ORT MLP slices is also slower than monolithic full ORT prefill, so a "safe single-layer RKNN" mode is not a viable performance path.
- RKNN precision/build-flag probes do not currently solve the drift:
  - `bf16` block0 `ln2_mlp.s320` builds on WSL2, but RK3576 runtime logs `input dtype is undefine` and `failed to submit`, returns all-zero output, and takes about `6400 ms` for one island. The probe now treats these stderr runtime errors as failures even when the process return code is 0.
  - `tf32` block0 build fails in toolkit with `Can not support request type: tfloat32`.
  - FP16 `optimization_level=0` for block0 is numerically identical to the existing opt3 artifact in sampler-boundary debug: `4/16` first-frame mismatches, `rel_l2=0.000954`, `cosine=0.99999946`, and the same TopK-missing counts. Lowering RKNN optimization level is not a quality fix for this island.
- Splitting `ln2` back to ORT and keeping only the MLP MatMul/GELU island on RKNN helps but still does not produce a production-safe composite:
  - `moss_block*_ln2.s320.onnx` + `moss_block*_mlp.s320.fp16.rk3576.rknn` all build, transfer, and run on RK3576. Runtime probe for 12 MLP-only RKNNs: `12/12 OK`, per-island inference roughly `17-40 ms`.
  - Single-layer first-frame scan with `--mlp-split mlp_only` shows layers `0,2,5,7,10` can individually produce `0/16` first-frame mismatches; layers `1,3,4,6,9,11` produce `9/16`, and layer `8` produces `2/16`.
  - Four-frame single-layer scan narrows the stable set to layers `0,2,5,10`; each has token mismatches `[0,0,0,0]`. Layer `7` diverges after decode with `[0,8,13,15]`.
  - The stable single-layer results are not additive. Combining `0,2,5,10` yields token mismatches `[3,15,16,16]`, and combining `0,2,5,7,10` yields `[3,15,16,16]`. The first-frame drift is lower than all-layer `ln2_mlp` (`rel_l2≈0.00065` vs `0.00235`), but sampler tokens still change. Therefore MLP-only split is useful diagnostic evidence, not a production default.
  - `moss_ort` now exposes this as an explicit experimental backend mode through `MOSS_ORT_HYBRID_SPLIT=mlp_only`, `MOSS_ORT_HYBRID_LAYERS=<set>`, and optional `MOSS_ORT_HYBRID_RKNN_DIR=<dir>`. RK3576 backend smoke with `mlp_only` layer `0` and strict hybrid loads successfully and streams two chunks, but TTFA is about `2256 ms` (`hybrid_prefill_ms≈2048`). This is slower than the production full-ORT fallback because the split runner still pays ORT attention plus ORT `ln2_mlp` slices for the other 11 layers. Keep it as a service-level experiment hook, not a production profile.

This moves the quality root cause from "unknown roundtrip regression" to a concrete sampler-boundary issue: full ORT slice composition is correct after KV trimming, but RKNN FP16 MLP output drift is large enough to change stochastic local-code candidate sets. Future RKNN work must prove token-level sampler parity and TopK/CDF stability, not just hidden cosine.

FastAPI service smoke with the current Junhao/314 full ORT fallback is covered by the unified service verifier:

```bash
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_service_streaming.py \
  --base-url http://127.0.0.1:8623 \
  --ws-url ws://127.0.0.1:8623/dialogue \
  --text 你好 \
  --expected-backend moss_ort \
  --expected-sample-rate 48000 \
  --min-payload-bytes 30720 \
  --max-tts-wall-ms 2000 \
  --max-dialogue-wall-ms 2000 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --json-out /tmp/moss_service_streaming_junhao314_ttfa.json
```

Verifier result:

```text
GET /health -> {"tts":true,"tts_backend":"moss_ort","streaming_tts":true,"asr":false,...}
POST /tts/stream -> 30724 bytes
stream header sample_rate=48000
payload_bytes=30720
int16_samples=15360
tts_stream.header_ms=46.618
tts_stream.first_payload_ms=1157.435
tts_stream.wall_ms=1400.970
WS /dialogue {"text":"你好"} -> sample_rate=48000, binary_chunks=3, payload_bytes=30720, done=true
dialogue.first_payload_ms=1146.298
dialogue.wall_ms=1375.225
max_tts_first_payload_ms=1500
max_dialogue_first_payload_ms=1500
max_tts_wall_ms=2000
max_dialogue_wall_ms=2000
gates.passed=true
```

That payload is two 3840-frame stereo chunks after the 4-byte sample-rate header.

The three binary messages are the sample-rate header plus two 80 ms stereo PCM chunks. This validates that dialogue mode now preserves backend-level streaming instead of buffering a full WAV per sentence.

Self-contained profile verifier after ORT and hybrid manifest validation were added. This command starts uvicorn, waits for `/health`, runs `/tts/stream` and `/dialogue`, writes JSON, and always stops the service process. For low-latency dialogue, first-payload time is the production gate; full stream wall time is recorded but may be disabled with `--max-*-wall-ms 0` when verifying production generation length.

```bash
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_service_profile.py \
  --repo-root /home/cat/rkvoice-stream \
  --port 8624 \
  --text 你好 \
  --expected-backend moss_ort \
  --expected-sample-rate 48000 \
  --min-payload-bytes 30720 \
  --max-tts-wall-ms 2000 \
  --max-dialogue-wall-ms 2000 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --require-manifest-validated \
  --expected-voice Junhao \
  --expected-seed 314 \
  --expected-manifest moss-ort-manifest.json \
  --json-out /tmp/moss_service_profile_manifest_gate.json \
  --log-file /tmp/moss_service_profile_manifest_gate_uvicorn.log \
  --set-env TTS_BACKEND=moss_ort \
  --set-env ASR_BACKEND=disabled \
  --set-env MOSS_ORT_MODEL_DIR=/home/cat/moss-onnx-baseline \
  --set-env MOSS_ORT_MANIFEST=moss-ort-manifest.json \
  --set-env MOSS_ORT_THREADS=6 \
  --set-env MOSS_ORT_MAX_NEW_FRAMES=2 \
  --set-env MOSS_ORT_CODEC_STREAMING=1 \
  --set-env MOSS_ORT_CACHE_VOICE_PREFIX=0 \
  --set-env MOSS_ORT_VOICE=Junhao \
  --set-env MOSS_ORT_SEED=314 \
  --set-env MOSS_ORT_WARMUP_TEXT=你好 \
  --set-env MOSS_ORT_HYBRID_RKNN=0
```

Result on RK3576:

```text
GET /health -> {"tts":true,"tts_backend":"moss_ort","streaming_tts":true,"asr":false,...}
health.tts_info.profile -> {"voice":"Junhao","seed":314,"threads":6,"prefill_seq":0,"max_new_frames":2,"codec_streaming":true}
health.tts_info.manifest -> {"name":"moss-ort-manifest.json","validated":true,"sha256":"8f2ed6769010afa21c26df467e1096de82d59e2c71c4615fee08d55cab5813b3","target_platform":"rk3576","required_artifacts":11}
POST /tts/stream -> sample_rate=48000, payload_bytes=30720
tts_stream.first_payload_ms=1105.659
tts_stream.wall_ms=1353.922
WS /dialogue -> sample_rate=48000, binary_chunks=3, payload_bytes=30720, done=true
dialogue.first_payload_ms=1122.068
dialogue.wall_ms=1368.858
require_manifest_validated=true
expected_voice=Junhao
expected_seed=314
expected_manifest=moss-ort-manifest.json
gates.passed=true
```

Production-length service verifier with `MOSS_ORT_MAX_NEW_FRAMES=20` keeps the same first-payload gate and disables full-response wall gates:

```bash
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_service_profile.py \
  --repo-root /home/cat/rkvoice-stream \
  --port 8628 \
  --text 你好 \
  --expected-backend moss_ort \
  --expected-sample-rate 48000 \
  --min-payload-bytes 30720 \
  --max-tts-wall-ms 0 \
  --max-dialogue-wall-ms 0 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --require-manifest-validated \
  --expected-voice Junhao \
  --expected-seed 314 \
  --expected-manifest moss-ort-manifest.json \
  --json-out /tmp/moss_service_profile_frames20_first_payload_gate.json \
  --set-env TTS_BACKEND=moss_ort \
  --set-env ASR_BACKEND=disabled \
  --set-env MOSS_ORT_MODEL_DIR=/home/cat/moss-onnx-baseline \
  --set-env MOSS_ORT_MANIFEST=moss-ort-manifest.json \
  --set-env MOSS_ORT_THREADS=6 \
  --set-env MOSS_ORT_MAX_NEW_FRAMES=20 \
  --set-env MOSS_ORT_CODEC_STREAMING=1 \
  --set-env MOSS_ORT_CACHE_VOICE_PREFIX=0 \
  --set-env MOSS_ORT_VOICE=Junhao \
  --set-env MOSS_ORT_SEED=314 \
  --set-env MOSS_ORT_WARMUP_TEXT=你好 \
  --set-env MOSS_ORT_HYBRID_RKNN=0
```

Result:

```text
health.tts_info.profile.max_new_frames=20
POST /tts/stream -> payload_bytes=307200, first_payload_ms=1150.437, wall_ms=5351.407
WS /dialogue -> binary_chunks=21, payload_bytes=307200, first_payload_ms=1169.181, wall_ms=5547.994
max_tts_wall_ms=0
max_dialogue_wall_ms=0
gates.passed=true
```

Unified production profile verifier on RK3576:

```bash
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_production_profile.py \
  --repo-root /home/cat/rkvoice-stream \
  --model-dir /home/cat/moss-onnx-baseline \
  --manifest moss-ort-manifest.json \
  --out-dir /tmp/moss_production_profile_junhao314 \
  --roundtrip-out-dir /tmp/moss_production_profile_junhao314_roundtrip \
  --json-out /tmp/moss_production_profile_junhao314.json \
  --port 8625 \
  --voice Junhao \
  --seed 314 \
  --threads 6 \
  --prefill-threads 8 \
  --decode-threads 5 \
  --codec-threads 5 \
  --prefill-seq 0 \
  --warmup-text 你好 \
  --roundtrip-max-new-frames 20 \
  --max-tts-wall-ms 0 \
  --max-dialogue-wall-ms 0 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --max-avg-cer 0.5 \
  --max-cer 1.0 \
  --min-rms 0.02 \
  --max-roundtrip-ttfa-ms 1500 \
  --max-codec-ms 120 \
  --log-file /tmp/moss_production_profile_junhao314_uvicorn.log
```

Unified result:

```text
summary.passed=true
checks.artifact_manifest=true
checks.service_streaming=true
checks.roundtrip_quality=true
artifact required_artifacts=11
service tts_stream.first_payload_ms=1112.957
service tts_stream.wall_ms=1397.816
service dialogue.first_payload_ms=1133.022
service dialogue.wall_ms=1370.278
roundtrip avg_cer=0.462121
roundtrip max_cer=0.75
roundtrip min_rms=0.085809
roundtrip max_ttfa_ms=1110
roundtrip max_codec_ms=76.933
```

Earlier `codec_batch_frames=4` production profile verifier rerun after tightening
the roundtrip TTFA gate to `1500 ms` and adding the dialogue chunk-cadence gate:

```bash
PYTHONPATH=/home/cat/rkvoice-stream \
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_production_profile.py \
  --repo-root /home/cat/rkvoice-stream \
  --model-dir /home/cat/moss-onnx-baseline \
  --manifest moss-ort-manifest.json \
  --asr-model-dir /home/cat/sherpa-onnx-paraformer-zh-2023-09-14 \
  --out-dir /tmp/moss_production_profile_current \
  --roundtrip-out-dir /tmp/moss_production_profile_current_roundtrip \
  --port 8624 \
  --voice Junhao \
  --seed 314 \
  --threads 6 \
  --prefill-threads 8 \
  --decode-threads 5 \
  --codec-threads 5 \
  --codec-batch-frames 4 \
  --prefill-seq 0 \
  --warmup-text 你好 \
  --service-max-new-frames 20 \
  --roundtrip-max-new-frames 20 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --max-roundtrip-ttfa-ms 1500 \
  --min-backend-audio-frames 20 \
  --max-backend-ttfa-ms 1500 \
  --max-backend-prefill-ms 1200 \
  --max-codec-ms 170 \
  --min-dialogue-binary-chunks 7 \
  --json-out /tmp/moss_production_profile_current.json
```

```text
summary.passed=true
artifact_manifest=true
service_streaming=true
roundtrip_quality=true
artifact required_artifacts=11
health profile: voice=Junhao seed=314 threads=6 session_threads={prefill:8,decode:5,sampler:6,codec:5} prefill_seq=0 max_new_frames=20 codec_streaming=true codec_batch_frames=4
service /tts/stream first_payload_ms=914.381 payload_bytes=307200 wall_ms=4362.068
service /dialogue first_payload_ms=1015.204 payload_bytes=307200 wall_ms=4543.464 binary_chunks=7
service gates: expected_codec_batch_frames=4 min_dialogue_binary_chunks=7
roundtrip avg_cer=0.462121 max_cer=0.75 min_rms=0.085809
roundtrip max_ttfa_ms=975 max_codec_ms=152.960
manifest sha256=8f2ed6769010afa21c26df467e1096de82d59e2c71c4615fee08d55cab5813b3
```

Earlier `codec_batch_frames=4` production profile verifier rerun after adding
the live runtime gate (`require_production_runtime=true`) on RK3576:

```bash
PYTHONPATH=/home/cat/rkvoice-stream \
/home/cat/rknn-venv/bin/python models/tts/moss/verify_moss_production_profile.py \
  --repo-root /home/cat/rkvoice-stream \
  --model-dir /home/cat/moss-onnx-baseline \
  --manifest moss-ort-manifest.json \
  --asr-model-dir /home/cat/sherpa-onnx-paraformer-zh-2023-09-14 \
  --out-dir /tmp/moss_production_profile_runtime_gate \
  --roundtrip-out-dir /tmp/moss_production_profile_runtime_gate_roundtrip \
  --port 8629 \
  --voice Junhao \
  --seed 314 \
  --threads 6 \
  --prefill-threads 8 \
  --decode-threads 5 \
  --codec-threads 5 \
  --codec-batch-frames 4 \
  --prefill-seq 0 \
  --warmup-text 你好 \
  --service-max-new-frames 20 \
  --roundtrip-max-new-frames 20 \
  --max-tts-first-payload-ms 1500 \
  --max-dialogue-first-payload-ms 1500 \
  --max-roundtrip-ttfa-ms 1500 \
  --max-codec-ms 170 \
  --min-dialogue-binary-chunks 7 \
  --json-out /tmp/moss_production_profile_runtime_gate.json
```

```text
summary.passed=true
artifact_manifest=true
service_streaming=true
roundtrip_quality=true
artifact required_artifacts=11
health runtime: voice=Junhao seed=314 session_threads={prefill:8,decode:5,sampler:6,codec:5}
health runtime: codec_streaming=true codec_full_loaded=false codec_batch_frames=4 codec_async=false cache_voice_prefix=false hybrid.enabled=false
service /tts/stream first_payload_ms=1035.733 payload_bytes=307200 wall_ms=4474.047
service /dialogue first_payload_ms=1062.422 payload_bytes=307200 wall_ms=4652.597 binary_chunks=7
service gates: require_manifest_validated=true require_production_runtime=true expected_codec_batch_frames=4 min_dialogue_binary_chunks=7
roundtrip avg_cer=0.462121 max_cer=0.75 min_rms=0.085809
roundtrip max_ttfa_ms=982 max_codec_ms=142.174
manifest sha256=8f2ed6769010afa21c26df467e1096de82d59e2c71c4615fee08d55cab5813b3
```

Backend-stage profiler for the same ORT production runtime settings:

```text
PRELOAD_MS=16959.035
summary gates passed=true
audio_frames=20 chunks=6 total_samples=76800 wall_ms=4630.268
ttfa_ms=949
prefill_ms=751.676
first sampler_ms=103.965
first codec_ms=90.848
max_decode_ms=67.685
max_sampler_ms=103.965
max_codec_ms=165.911
chunk_shapes=[[3840,2],[15360,2],[15360,2],[15360,2],[15360,2],[11520,2]]
```

After skipping the unused full-codec ONNX session when streaming codec is
available, the same backend-stage gate still passes and startup improves:

```text
PRELOAD_MS=14503.987
summary gates passed=true
audio_frames=20 chunks=6 total_samples=76800 wall_ms=4575.353
ttfa_ms=939
prefill_ms=750.038
first sampler_ms=93.809
first codec_ms=92.915
max_decode_ms=78.427
max_sampler_ms=93.809
max_codec_ms=166.223
```

Production `moss_ort` now loads `moss_audio_tokenizer_decode_step.onnx` for
streaming codec and does not load `moss_audio_tokenizer_decode_full.onnx` unless
`MOSS_ORT_LOAD_FULL_CODEC=1` is explicitly set or streaming codec is disabled.
The full-codec artifact still remains in the manifest so offline/fallback
profiles are hash-validated, but low-latency service startup avoids the unused
session.

Use `models/tts/moss/smoke_moss_hybrid_backend.py --json-out ...` as the
backend-stage profiling gate. For batched codec output, gate generation length
with `--min-audio-frames 20`, not raw chunk count: one HTTP/WebSocket payload can
contain three 80 ms codec frames when `codec_batch_frames=3`.

The verifier now also records `dialogue.payload_chunk_times_ms` and gates
`dialogue.max_payload_gap_ms` when `--max-dialogue-payload-gap-ms` is set.
The RK3576 service-only cadence gate with the same production runtime settings
passed `--max-dialogue-payload-gap-ms 1500`:

```text
service /tts/stream first_payload_ms=947.473 payload_bytes=307200 wall_ms=4397.272
service /dialogue first_payload_ms=1006.185 payload_bytes=307200 wall_ms=4536.868 binary_chunks=7
service /dialogue payload_chunk_times_ms=[1006.185,1770.395,2516.354,3230.623,3985.080,4535.570]
service /dialogue max_payload_gap_ms=764.210
service gates: max_dialogue_payload_gap_ms=1500 passed=true
```

Keep this gate in production reruns so acceptance evidence proves sustained
streaming cadence after first audio, not just a fast first payload.

Earlier `codec_batch_frames=4` service runtime gate after skipping the unused
full-codec session:

```text
health runtime: codec_streaming=true codec_full_loaded=false codec_batch_frames=4 codec_async=false cache_voice_prefix=false hybrid.enabled=false
service /tts/stream first_payload_ms=1020.928 payload_bytes=307200 wall_ms=4558.769
service /dialogue first_payload_ms=986.985 payload_bytes=307200 wall_ms=4443.763 binary_chunks=7
service /dialogue max_payload_gap_ms=748.691
service gates: require_manifest_validated=true require_production_runtime=true expected_codec_batch_frames=4 min_dialogue_binary_chunks=7 passed=true
```

Earlier `codec_batch_frames=4` full production profile after the same full-codec
skip also passed artifact, service, and ASR roundtrip gates before the
backend-stage jitter gate was added:

```text
summary.passed=true
artifact_manifest=true
service_streaming=true
roundtrip_quality=true
artifact required_artifacts=11
health runtime: voice=Junhao seed=314 session_threads={prefill:8,decode:5,sampler:6,codec:5}
health runtime: codec_streaming=true codec_full_loaded=false codec_batch_frames=4 codec_async=false cache_voice_prefix=false hybrid.enabled=false
service /tts/stream first_payload_ms=917.558 payload_bytes=307200 wall_ms=4386.835
service /dialogue first_payload_ms=1028.797 payload_bytes=307200 wall_ms=4475.118 binary_chunks=7
service /dialogue max_payload_gap_ms=726.902
roundtrip avg_cer=0.462121 max_cer=0.75 min_rms=0.085809
roundtrip max_ttfa_ms=947 max_codec_ms=149.428
manifest sha256=8f2ed6769010afa21c26df467e1096de82d59e2c71c4615fee08d55cab5813b3
```

Latest production profile after adding the backend-stage gate uses
`codec_batch_frames=3`. A four-frame rerun passed service and roundtrip but
failed the new backend-stage codec jitter gate (`max_codec_ms=210.116 > 170`).
The three-frame profile passed all four production checks:

```text
summary.passed=true
artifact_manifest=true
service_streaming=true
backend_stage=true
roundtrip_quality=true
health runtime: codec_streaming=true codec_full_loaded=false codec_batch_frames=3 codec_async=false cache_voice_prefix=false hybrid.enabled=false
service /tts/stream first_payload_ms=988.302 payload_bytes=307200 wall_ms=4561.744
service /dialogue first_payload_ms=1026.240 payload_bytes=307200 wall_ms=4682.886 binary_chunks=9
service /dialogue max_payload_gap_ms=599.983
backend_stage ttfa_ms=907 prefill_ms=717.953 audio_frames=20 chunks=8 wall_ms=4639.865
backend_stage max_decode_ms=69.139 max_sampler_ms=98.395 max_codec_ms=130.455
roundtrip avg_cer=0.462121 max_cer=0.75 min_rms=0.085809
roundtrip max_ttfa_ms=1056 max_codec_ms=125.925
json=/tmp/moss_production_profile_codec_batch3.json
```

Current-code rerun of the same three-frame production profile on RK3576 also
passes all four gates after the `moss_rknn` manifest quality-status guard:

```text
summary.passed=true
artifact_manifest=true
service_streaming=true
backend_stage=true
roundtrip_quality=true
health runtime: codec_streaming=true codec_full_loaded=false codec_batch_frames=3 codec_async=false cache_voice_prefix=false hybrid.enabled=false
service /tts/stream first_payload_ms=1031.002 payload_bytes=307200 wall_ms=4506.959
service /dialogue first_payload_ms=1075.154 payload_bytes=307200 wall_ms=4700.098 binary_chunks=9
service /dialogue max_payload_gap_ms=574.937
backend_stage ttfa_ms=908 prefill_ms=735.267 audio_frames=20 chunks=8 wall_ms=4641.572
backend_stage max_decode_ms=71.105 max_sampler_ms=90.003 max_codec_ms=143.680
roundtrip avg_cer=0.462121 max_cer=0.75 min_rms=0.085809
roundtrip max_ttfa_ms=952 max_codec_ms=118.615
device_json=/tmp/moss_production_profile_current_rerun.json
repo_json=docs/evidence/moss/rk3576-moss-ort-production-current-rerun.json
repo_roundtrip_json=docs/evidence/moss/rk3576-moss-ort-production-current-rerun-roundtrip.json
```

After adding `streaming_stats` observability, the RK3576 service-level verifier
was rerun with `/health` before/after gates enabled:

```text
gates.passed=true
health.streaming_stats before: requests=0 completed=0 errors=0 active=0 chunks=0
health.streaming_stats after: requests=2 completed=2 errors=0 active=0 chunks=16
service /tts/stream first_payload_ms=1002.303 payload_bytes=307200 wall_ms=4554.166
service /dialogue first_payload_ms=1050.970 payload_bytes=307200 wall_ms=4657.619 binary_chunks=9
service /dialogue max_payload_gap_ms=585.431
repo_json=docs/evidence/moss/rk3576-moss-service-streaming-stats-profile.json
repo_log=docs/evidence/moss/rk3576-moss-service-streaming-stats-uvicorn.log
```

Historical deployment path preflight failed for the canonical checked-in
production config because `/opt/tts/models/moss-tts-nano-onnx` was not present:

```text
python models/tts/moss/run_moss_production_server.py --dry-run --validate-artifacts
passed=false
ort_config=true
ort_artifacts=false
error=tts.model_dir does not exist: /opt/tts/models/moss-tts-nano-onnx
repo_json=docs/evidence/moss/rk3576-moss-production-server-opt-artifact-gate.json
```

That JSON includes remediation commands under `checks.ort_artifacts.remediation`:
a dry-run command for `prepare_moss_ort_deployment.py`, and the explicit
`--execute --confirm INSTALL_MOSS_ORT_DEPLOYMENT` command required to write the
canonical symlink under `/opt`.
The release audit has the same optional deployment artifact gate:

```text
python models/tts/moss/audit_moss_release.py --validate-ort-artifacts
passed=false
ort_config=true
ort_evidence=true
roundtrip_evidence=true
production_entrypoint=true
rknn_candidate_config=true
ort_artifacts=false
error=tts.model_dir does not exist: /opt/tts/models/moss-tts-nano-onnx
repo_json=docs/evidence/moss/rk3576-moss-release-audit-ort-artifact-gate.json
```

The prepared canonical-layout dry-run validates `/home/cat/moss-onnx-baseline`
and plans a symlink instead of copying the large ONNX bundle:

```text
python models/tts/moss/prepare_moss_ort_deployment.py \
  --source /home/cat/moss-onnx-baseline \
  --destination /opt/tts/models/moss-tts-nano-onnx
passed=true
execute=false
source_manifest.required_artifacts=11
commands:
  mkdir -p /opt/tts/models
  ln -s /home/cat/moss-onnx-baseline /opt/tts/models/moss-tts-nano-onnx
repo_json=docs/evidence/moss/rk3576-moss-ort-deployment-dry-run.json
```

Execution is intentionally gated by `--execute --confirm
INSTALL_MOSS_ORT_DEPLOYMENT` because it writes under `/opt`.

The canonical `/opt` symlink has now been installed on the RK3576 board and the
artifact gate passes:

```text
python models/tts/moss/prepare_moss_ort_deployment.py \
  --source /home/cat/moss-onnx-baseline \
  --destination /opt/tts/models/moss-tts-nano-onnx \
  --execute \
  --confirm INSTALL_MOSS_ORT_DEPLOYMENT
passed=true
execute=true
deployed=true
mode=symlink
source_manifest.required_artifacts=11
repo_json=docs/evidence/moss/rk3576-moss-ort-deployment-execute.json

python models/tts/moss/audit_moss_release.py --validate-ort-artifacts
passed=true
production_entrypoint=true
ort_artifacts=true
model_dir=/opt/tts/models/moss-tts-nano-onnx
required_artifacts=11
repo_json=docs/evidence/moss/rk3576-moss-release-audit-ort-artifact-pass.json

python models/tts/moss/run_moss_production_server.py --dry-run --validate-artifacts
passed=true
service.command="/home/cat/rknn-venv/bin/python -m uvicorn rkvoice_stream.app.server:app --host 0.0.0.0 --port 8621"
repo_json=docs/evidence/moss/rk3576-moss-production-server-opt-artifact-pass.json
```

Canonical-path service verification also passed the production streaming gates:

```text
health.tts_info.model_dir=/opt/tts/models/moss-tts-nano-onnx
health.tts_info.manifest.validated=true
health.tts_info.profile.codec_streaming=true
health.tts_info.profile.codec_full_loaded=false
health_after.streaming_stats: requests=2 completed=2 errors=0 active=0 chunks=16
service /tts/stream first_payload_ms=941.344 payload_bytes=307200 wall_ms=4498.332
service /dialogue first_payload_ms=1036.769 payload_bytes=307200 wall_ms=4642.550 binary_chunks=9
service /dialogue max_payload_gap_ms=588.320
repo_json=docs/evidence/moss/rk3576-moss-canonical-profile.json
repo_log=docs/evidence/moss/rk3576-moss-canonical-profile-uvicorn.log
```

The audited production entrypoint has also been verified end-to-end. This starts
`run_moss_production_server.py`, lets the runner perform release/artifact
preflight, then verifies the live service:

```text
python models/tts/moss/verify_moss_production_entrypoint.py \
  --config configs/rk3576-moss-ort-stream.yaml \
  --host 127.0.0.1 \
  --port 8629 \
  --python /home/cat/rknn-venv/bin/python \
  --require-manifest-validated \
  --expected-voice Junhao \
  --expected-seed 314 \
  --expected-manifest moss-ort-manifest.json \
  --expected-codec-batch-frames 3 \
  --require-production-runtime \
  --min-dialogue-binary-chunks 7
preflight.passed=true
preflight.ort_artifacts=true
service /tts/stream first_payload_ms=942.947 payload_bytes=307200 wall_ms=4451.263
service /dialogue first_payload_ms=1051.573 payload_bytes=307200 wall_ms=4662.182 binary_chunks=9
service /dialogue max_payload_gap_ms=574.605
health_after.streaming_stats: requests=2 completed=2 errors=0 active=0 chunks=16
repo_preflight=docs/evidence/moss/rk3576-moss-production-entrypoint-preflight.json
repo_json=docs/evidence/moss/rk3576-moss-production-entrypoint-profile.json
repo_log=docs/evidence/moss/rk3576-moss-production-entrypoint.log
```

This profile is now part of `audit_moss_release.py` as the
`production_entrypoint` check, so a stale runner, wrong config, non-canonical
model path, unvalidated manifest, disabled streaming codec, hybrid leakage, or
streaming runtime error fails the default release audit instead of living only
as a side verification artifact.

Uvicorn signal handling was also verified after removing the app-level
`SIGTERM`/`SIGINT` override. RK3576 short-start smoke now shuts down cleanly
through FastAPI shutdown without `SystemExit`, `Traceback`, or `ERROR` log
entries:

```text
repo_log=docs/evidence/moss/rk3576-uvicorn-signal-smoke.log
```

Conclusion: the ORT path is the current production fallback for RK3576 MOSS because it is streamable and matches official token/audio quality. The hybrid ORT+RKNN path is a performance experiment only: it preserves streaming and lowers TTFA, but it fails ASR roundtrip quality. `moss_ort` rejects text without `sentencepiece` unless `MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK=1` is explicitly set for smoke tests.

Tokenizer install note: RK3576 DNS resolution failed for both online `pip install sentencepiece` and `apt-get install python3-sentencepiece` (`域名解析暂时失败`). The working path was downloading `sentencepiece-0.2.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` on the control machine, pushing it over LAN, and installing it into `/home/cat/rknn-venv` with `pip install --no-index`.

RKNN probe status on RK3576:

- Use `models/tts/moss/probe_moss_rknn_runtime.py` for runtime probes. It runs each RKNN in a child process and writes JSON so `inference()` SIGSEGV does not kill the whole sweep:

```bash
PYTHONPATH=/home/cat/rkvoice-stream \
/home/cat/rknn-venv/bin/python models/tts/moss/probe_moss_rknn_runtime.py \
  --timeout 25 \
  --pass-through auto \
  --json-out /home/cat/moss-rknn-probe/runtime_probe_auto_phase.json \
  /home/cat/moss-rknn-probe/min_layernorm.fp16.rk3576.rknn \
  /home/cat/moss-rknn-probe/moss_tts_prefill.s32.fp16.rk3576.cumsumfix.rknn \
  /home/cat/moss-rknn-probe/moss_tts_decode_step.p1.fp16.rk3576.rknn \
  /home/cat/moss-rknn-probe/moss_tts_local_fixed_sampled_frame.fp16.rk3576.rknn
```

- Probe baseline: `min_layernorm.fp16.rk3576.rknn` loads, initializes, and runs in about `3.3-5.9 ms`; output is finite. This proves the RKNN Lite install and NPU runtime are not globally broken.
- Original `moss_tts_prefill.s32.fp16.rk3576.rknn` converted, loaded, and initialized, but inference failed with `Unsupport CPU op: CumSum` and process exit 139.
- `models/tts/moss/convert_moss_rknn.py` now rewrites fixed-bucket prefill `CumSum` into a float `MatMul` prefix-sum equivalent before RKNN conversion.
- `models/tts/moss/convert_moss_rknn.py` also supports `--prefill-crop-output <tensor>` for RKNN crash probes.
- Fixed `moss_tts_prefill.s32.fp16.rk3576.rknn` converted successfully. Build log has no `CumSum`, `No lowering`, or CPU fallback warning.
- Fixed prefill copied to `cat-remote:/home/cat/moss-rknn-probe/moss_tts_prefill.s32.fp16.rk3576.cumsumfix.rknn`; md5 `94c1cfc3332301a431e834acea8c4bfd`.
- On RK3576, fixed prefill loads in about `180-217 ms` and `init_runtime` completes in about `559 ms`, but dies during `inference()` with SIGSEGV. `pass-through=auto` and `pass-through=none` both fail.
- Single-output `global_hidden` prefill still dies during `inference()` with SIGSEGV after successful load/init, so the failure is not caused by the 24 KV outputs.
- Early crop `/Add_15_output_0` converts to a 50 MB RKNN, but dies during `inference()` with SIGSEGV even when probed as a single `input_ids[1,32,17]` input.
- Earliest token embedding crop `/wte/Gather_output_0` converts to a 25 MB RKNN, but dies during `inference()` with SIGSEGV even when probed as a single `input_ids[1,32,17]` input. Avoid putting MOSS token embedding `Gather` inside the RKNN graph until a different conversion/runtime path is proven.
- A synthetic one-input LayerNormalization RKNN runs on RK3576 when float input is not pass-through. For mixed inputs, use float tensors through the default input path and only pass-through integer tensors; using pass-through for float tensors can itself cause exit 139.
- `moss_tts_prefill.s32.suffix_from_Add15.fp16.rk3576.rknn` (`hidden[1,32,768] + attention_mask[1,32] -> global_hidden`) converts to about 168 MB but dies during `inference()` with SIGSEGV after successful load/init. The remaining failure is therefore in the MOSS position/attention-mask/transformer prefix combination, not only in token embedding.
- `moss_tts_prefill.s32.suffix_Add15_to_ln1.fp16.rk3576.rknn` also dies during `inference()` with SIGSEGV when probed as `hidden[1,32,768] + attention_mask[1,32]`.
- `moss_tts_local_fixed_sampled_frame.fp16.rk3576.rknn` converts to about 341 MB and transfers over LAN, but dies during sampler `inference()` with SIGSEGV under both `pass-through=auto` and `pass-through=none`. This is not production usable on RK3576 as a monolithic RKNN.
- `moss_tts_decode_step.p1.fp16.rk3576.rknn` converts to about 217 MB and loads/initializes on RK3576, but dies during `inference()` with SIGSEGV under both `pass-through=auto` and `pass-through=none`.
- `codec_decode_step.f1.fp16.rk3576.rknn` does not currently build with RKNN toolkit 2.3.0/2.3.x. `optimization_level` 3, 2, 1, and 0 all fail in RKNN MatMul fuse with `IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed`.

The current codebase has a machine-readable RK3576 runtime probe for the
monolithic non-sampler candidates:

```text
python models/tts/moss/probe_moss_rknn_runtime.py \
  --timeout 45 \
  --pass-through auto \
  --json-out /tmp/moss_monolithic_prefill_decode_rknn_probe.json \
  /home/cat/moss-rknn-probe/moss_tts_prefill.s32.fp16.rk3576.cumsumfix.rknn \
  /home/cat/moss-rknn-probe/moss_tts_decode_step.p1.fp16.rk3576.rknn
summary: total=2 ok=0 crash=2 timeout=0 missing=0
prefill: load_ret=0 init_ret=0 status=CRASH signal=SIGSEGV
decode: load_ret=0 init_ret=0 status=CRASH signal=SIGSEGV
repo_json=docs/evidence/moss/rk3576-moss-monolithic-prefill-decode-rknn-probe.json
```

This confirms that full prefill/decode-step RKNN remains disqualified even
though the artifacts load and initialize. Future non-sampling acceleration must
stay at narrower verified islands or use a different export/runtime strategy;
do not promote these monolithic artifacts into the `moss_rknn` production
manifest.

Validated RKNN islands on RK3576:

- `models/tts/moss/extract_moss_rknn_island.py` extracts small ONNX islands from `moss_tts_prefill.onnx` and optionally converts them to RKNN. The first useful class is pure-float transformer MLP/projection islands, avoiding token `Gather`, rotary `Sin/Cos`, dynamic mask creation, and full attention.
- `models/tts/moss/verify_moss_rknn_island_parity.py` compares each island against ONNX Runtime with deterministic float input and records latency.

Commands used on WSL2:

```bash
cd /home/harve/project/rkvoice-stream
/home/harve/rknn-build/.venv/bin/python models/tts/moss/extract_moss_rknn_island.py \
  --onnx /home/harve/models/moss-onnx-bundle-paged-fp16/MOSS-TTS-Nano-100M-ONNX/moss_tts_prefill.onnx \
  --out-dir /home/harve/models/moss-rknn-islands \
  --preset ln2_mlp \
  --layer 0 \
  --seq-len 320 \
  --convert-rknn \
  --target rk3576 \
  --precision fp16 \
  --force
```

For new device-side island extraction/conversion runs, write under the prepared
RKNN workspace and enable the same preflight used by full conversion:

```bash
python models/tts/moss/extract_moss_rknn_island.py \
  --onnx /opt/tts/models/moss-tts-nano-onnx/moss_tts_prefill.onnx \
  --out-dir /mnt/rknn-workspace/moss-rknn-workspace/moss-rknn-islands \
  --preset ln2_mlp \
  --layer 0 \
  --seq-len 320 \
  --convert-rknn \
  --require-rknn-workspace \
  --rknn-workspace /mnt/rknn-workspace/moss-rknn-workspace \
  --target rk3576 \
  --precision fp16
```

RK3576 probe/parity results, all with input shape `[1,32,768]`:

```text
block0_cproj:
  rknn_infer_avg=2.167 ms, ort_avg=1.515 ms
  rel_l2=0.000291, cosine=0.9999996, passed=true

block0_fc_in_act:
  rknn_infer_avg=3.027 ms, ort_avg=4.782 ms
  rel_l2=0.000586, cosine=0.9999972, passed=true

block0_mlp:
  rknn_infer_avg=3.883 ms, ort_avg=8.267 ms
  rel_l2=0.000515, cosine=0.9999992, passed=true

block0_ln2_mlp:
  rknn_infer_avg=2.468 ms, ort_avg=7.236 ms
  rel_l2=0.002392, cosine=0.9999987, passed=true
```

Artifacts on RK3576:

```text
/home/cat/moss-rknn-probe/moss_block0_cproj.s32.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/moss_block0_fc_in_act.s32.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/moss_block0_mlp.s32.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/moss_block0_ln2_mlp.s32.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/moss_block0_mlp.s320.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/moss_block0_ln2_mlp.s320.fp16.rk3576.rknn
/home/cat/moss-rknn-probe/parity_block0_*_timed.json
/home/cat/moss-rknn-probe/parity_block0_*_s320_timed.json
```

This is the first verified MOSS RKNN acceleration path on RK3576 and is now integrated into the streaming TTS backend as the `moss_ort` hybrid prefill mode. The net service-level latency win is real, but quality currently fails after the sampler. Do not ship it as default until roundtrip quality recovers.

Full-prompt-scale MLP island results with input shape `[1,320,768]` are more relevant to the current `Lingyu` full prompt path (`~300` rows):

```text
block0_mlp.s320:
  rknn_first=37.036 ms, rknn_avg=15.354 ms
  ort_first=68.211 ms, ort_avg=60.966 ms
  rel_l2=0.000520, cosine=0.9999976, passed=true

block0_ln2_mlp.s320:
  rknn_first=35.505 ms, rknn_avg=15.317 ms
  ort_first=65.693 ms, ort_avg=65.563 ms
  rel_l2=0.002237, cosine=0.9999973, passed=true

block0_cattn.s320:
  boundary=/ln_1/LayerNormalization_output_0 -> /c_attn/Add_output_0
  rknn_first=30.668 ms, rknn_avg=14.941 ms
  ort_first=26.588 ms, ort_avg=24.300 ms
  rel_l2=0.000292, cosine=0.9999942, passed=true
  runtime_probe=OK, output=[1,320,2304], artifact_size=3.6 MB

all 12 cattn.s320:
  build=OK for layers 0-11, artifact_size≈3.6 MB/layer
  runtime_probe: total=12, ok=12, crash=0, timeout=0
  parity: passed=true, max_rel_l2=0.000293, min_cosine=0.9999937
  sum_ort_avg=303.874 ms, sum_rknn_avg=185.025 ms, speedup=1.642x
```

This shows real prefill-scale speedups for deterministic projection/MLP islands.
`block0_ln2_mlp.s320` gives roughly `4x` for the MLP path, and the new
all-layer `cattn.s320` qkv projection set gives about `1.6x` steady-state
island speedup while avoiding the full attention slice crash. The theoretical upper
bound is meaningful because MLP/projection work repeats across 12 transformer
blocks, but the practical service-level gain depends on whether the hybrid
backend can avoid excessive CPU/RKNN tensor copies and NPU call overhead.
`block0_cproj.s32` was slower than ORT at small sequence length, so
projection-only islands still need per-boundary full-prompt probes instead of
blanket promotion.

All 12 `ln2_mlp.s320` islands now build, load, initialize, run, and pass parity on RK3576:

```text
runtime_probe_ln2_mlp_s320_all.json:
  total=12, ok=12, crash=0, timeout=0

parity_block*_ln2_mlp_s320_timed.json:
  all_passed=true
  sum_ort_avg=740.995 ms
  sum_rknn_avg=187.194 ms
  mean_rel_l2=0.002291
  min_cosine=0.9999961
```

Co-resident 12-model smoke in one Python process:

```text
RSS before load: ~31 MB
RSS after 12 load/init: ~339 MB
first sequential 12-layer inference: 261.4 ms
steady sequential 12-layer average after first: 203.02 ms
RSS after one sequential pass: ~388 MB
RSS after release: ~188 MB
```

This means memory is not the immediate blocker for a hybrid MLP path on the 8GB RK3576. The practical upper-bound MLP saving is roughly `740.995 - 203.02 = 537.975 ms` for a full-prompt-scale prefill if all 12 `ln2_mlp` islands are called sequentially from one process. That is not yet an end-to-end TTFA improvement because the current backend still uses monolithic ORT prefill; the next step is to split prefill around attention/MLP boundaries or implement a layer runner that can feed ORT attention outputs into RKNN MLP islands.

Conclusion: the WSL2 Jetson ONNX export is reusable and production service should start from ORT. The current RKNN artifacts are not production usable on RK3576 as monolithic graphs.

Transfer note: WSL2 and RK3576 are on the same LAN (`192.168.3.230` and `192.168.3.216`). Use direct fleet transfer for large RKNN files:

```bash
uv run --project ~/project/_hub python ~/project/_hub/fleet.py transfer \
  --dest-host 192.168.3.216 \
  wsl2-local:/path/model.rknn \
  cat-remote:/home/cat/moss-rknn-probe/model.rknn
```

Do not use `--relay` for these large model probes unless direct transfer fails.

The RK3576 board was rebooted after an RKNPU submit failure state, then the Qwen3 RKNN probe proved the device can run a streaming TTS pipeline again:

- Qwen3 prefill path loaded and ran after reboot.
- First audio chunk for `你好`: about 1515 ms, 3840 samples.
- Four streamed chunks completed in about 5721 ms.

This is only a device/NPU health probe. It is not MOSS production evidence.

## FC Split RKNN Probe, 2026-05-24

The latest split moved only the MLP `fc_out` projection to RKNN and kept
`ln2` + `fc_in_act` on ORT for the selected layers. This reduces the FP16
boundary drift versus `ln2_mlp` and `mlp_only`, but it is still experimental.

Runtime probe on RK3576:

```text
/home/cat/moss-rknn-fc-split-probe/moss_block*_fc_*.s320.fp16.rk3576.rknn
total=24, ok=24, crash=0, timeout=0
fc_in_act infer range: ~22.8-43.5 ms, output [1,320,3072]
fc_out infer range: ~53.9-61.5 ms, output [1,320,768]
```

Single-layer `fc_out_only` sampler parity, 4 generated frames, seed 314:

```text
pass: layers 0,1,4,5,6,10,11
fail at frame 3 with 2 token mismatches: layers 2,3,7,8,9
```

Layer-combination results show the usual sampler sensitivity: safe single
layers are not automatically additive.

```text
0,1                 pass [0,0,0,0]
0,1,4               pass [0,0,0,0]
0,1,4,5             fail [0,1,14,16]
0,1,4,5,6           pass [0,0,0,0]
0,1,4,5,6,10        fail [0,1,14,16]
0,1,4,5,6,10,11     fail [0,1,14,16]
10,11               pass [0,0,0,0]
0,10,11             pass [0,0,0,0]
```

The current experimental backend profile therefore uses
`hybrid_split=fc_out_only` and `hybrid_layers=0,1,4,5,6`. This is not a
production profile yet: the evidence above proves local sampler parity for a
short probe, not TTFA improvement or ASR roundtrip quality. The production
profile remains full ORT streaming.

Backend smoke after wiring `fc_out_only` into `moss_ort`:

```text
MOSS_ORT_HYBRID_SPLIT=fc_out_only
MOSS_ORT_HYBRID_LAYERS=0,1,4,5,6
PRELOAD_MS=18356.955
first chunk: [3840,2]
prefill_ms=2188.782
ttfa_ms=2408
2 chunks wall_ms=3010.463
```

This is functional but slower than the ORT production fallback measured earlier
(`~1.15s` first payload). It remains a verifier-backed experiment, not a
performance win.

Strict service-level verifier with the manifest-backed `fc_out_only` split,
`hybrid_layers=0,1,4,5,6`, production ORT session thread settings, streaming
codec batch size `4`, and `max_new_frames=20`:

```text
health hybrid: enabled=true strict=true split=fc_out_only layers=[0,1,4,5,6]
manifest: moss-hybrid-fc-out-manifest.json, required=36, production_default=false
/tts/stream first_payload_ms=2605.437 payload_bytes=199680 wall_ms=4835.846
/dialogue first_payload_ms=2551.434 payload_bytes=138240 wall_ms=4054.082 binary_chunks=4
/dialogue payload_chunk_times_ms=[2551.434,3317.751,4052.264]
/dialogue max_payload_gap_ms=766.317
gates passed=false:
  tts stream first_payload_ms=2605.437 exceeds 2500.000
  dialogue binary_chunks=4 below 7
  dialogue first_payload_ms=2551.434 exceeds 2500.000
```

This closes the current `fc_out_only` candidate for production promotion. It
has a stronger artifact contract now, but service TTFA is worse than the
manifest-verified ORT fallback and generated stream length/chunk count regresses
under the same `max_new_frames=20` service profile. Do not run a full ASR
roundtrip for this candidate unless a future split first restores service-level
TTFA and chunk-count gates.

Backend-stage profiler confirms the service failure is prefill-side, not codec:

```text
PRELOAD_MS=21653.631
summary gates passed=false
audio_frames=13 chunks=4 total_samples=49920 wall_ms=5177.434
ttfa_ms=2596
prefill_ms=2380.449
hybrid_prefill_ms=2377.855
first sampler_ms=126.489
first codec_ms=84.949
max_decode_ms=69.249
max_sampler_ms=126.489
max_codec_ms=158.912
gates failed:
  audio_frames=13 below 20
  ttfa_ms=2596.000 exceeds 1500.000
  prefill_ms=2380.449 exceeds 1200.000
```

Per-layer timing shows the selected RKNN `fc_out_only` islands are not faster
than the ORT MLP slices in this service composition. RKNN fc-out layers measured
about `99-164 ms` in the MLP path, while ORT `ln2_mlp` fallback layers measured
about `96-111 ms`. The extra ORT/RKNN boundary and NPU call overhead dominates
any local projection speedup, so continuing to add fc-out layers is not a
credible high-performance path.

## Attention Residual RKNN Probe, 2026-05-24

Because `fc_out_only` does not beat ORT TTFA, the next target was the larger
ORT attention cost. `attn_residual` was extracted as a two-input slice:

```text
inputs: hidden [1,S,768], attention_mask [1,S]
outputs: attention residual, present_key_0, present_value_0
```

The first build kept the original ONNX `CumSum`; RK3576 crashed on inference
with:

```text
Unsupport CPU op: CumSum in this librknnrt.so
signal=SIGSEGV
```

After reusing the fixed-bucket `CumSum -> Cast + MatMul(upper triangular ones)
+ Cast` surgery, build succeeded and the unsupported-op error disappeared, but
RK3576 still crashed at inference for both `s320` and `s32`:

```text
moss_block0_attn_residual.s320.fp16.rk3576.rknn:
  load_ret=0, init_ret=0, status=CRASH, signal=SIGSEGV
  pass_through auto/none/all all crash

moss_block0_attn_residual.s32.fp16.rk3576.rknn:
  load_ret=0, init_ret=0, status=CRASH, signal=SIGSEGV
```

Conclusion: `attn_residual` is not a viable RKNN coverage target on the current
RK3576 runtime. The next performance path should avoid this full attention
slice. `extract_moss_rknn_island.py` now includes a narrower `cattn` preset for
the fused qkv projection only (`/ln_1*/LayerNormalization_output_0 ->
/c_attn*/Add_output_0`), alongside the existing `cproj` preset. All 12
`cattn.s320` artifacts build, run on RK3576, pass parity, and are faster than
ORT in isolation. This still excludes mask/softmax/KV operators that are
implicated in the full attention-slice crashes. The next work item is backend
integration accounting: preload the 12 `cattn` RKNN contexts beside the existing
`ln2_mlp` contexts and measure whether the extra ORT/RKNN handoff preserves the
`~119 ms` isolated projection saving.

The integration accounting now passes on RK3576 with the same `Lingyu`/`你好`
`s320` prefill prompt:

```text
baseline hybrid, ln2_mlp RKNN only:
  gates_passed=true
  global_hidden rel_l2=0.002594, cosine=0.9999962
  kv_max_rel_l2=0.003296, kv_min_cosine=0.9999906
  full_prefill_target_ms=1674.329
  hybrid_prefill_ms=1135.396
  sum_attention_ort_ms=812.248
  sum_mlp_rknn_ms=233.545

cattn hybrid, ln1 ORT + cattn RKNN + attention suffix ORT + ln2_mlp RKNN:
  gates_passed=true
  global_hidden rel_l2=0.002633, cosine=0.9999968
  kv_max_rel_l2=0.003349, kv_min_cosine=0.9999907
  full_prefill_target_ms=1649.900
  hybrid_prefill_ms=1059.671
  sum_ln1_ort_ms=45.890
  sum_cattn_rknn_ms=229.367
  sum_attention_suffix_ort_ms=521.118
  sum_mlp_rknn_ms=198.936
```

The net prefill gain from adding all 12 `cattn` islands is therefore
`1135.396 / 1059.671 = 1.071x`, or about `75.7 ms` saved on this prompt.
This is real NPU acceleration, but much smaller than the isolated `cattn`
island speedup (`1.642x`) because the integrated runner still pays 12 ORT
`ln1` calls, 12 ORT attention suffix calls, and 12 additional ORT/RKNN tensor
handoffs. Treat this as a validated building block, not a production promotion
by itself. To materially beat the ORT streaming profile, the next split must
either reduce per-layer handoff overhead or move a larger stable attention/MLP
region onto NPU without reintroducing the `attn_residual` runtime crash.

The first coarser attention-side split is now verified: `ln1_cattn.s320`
combines the masked hidden input, `ln1`, and fused qkv projection in one RKNN
island:

```text
all 12 ln1_cattn.s320:
  build=OK for layers 0-11, artifact_size≈3.67 MB/layer
  runtime_probe: total=12, ok=12, crash=0, timeout=0
  parity: passed=true, max_rel_l2=0.001874, min_cosine=0.9999934
  sum_ort_avg=346.660 ms, sum_rknn_avg=188.159 ms, speedup=1.842x

ln1_cattn hybrid, ln1_cattn RKNN + attention suffix ORT + ln2_mlp RKNN:
  gates_passed=true
  global_hidden rel_l2=0.003849, cosine=0.9999923
  kv_max_rel_l2=0.004952, kv_min_cosine=0.9999846
  full_prefill_target_ms=1700.651
  hybrid_prefill_ms=979.765
  sum_ln1_cattn_rknn_ms=240.824
  sum_attention_suffix_ort_ms=485.855
  sum_mlp_rknn_ms=192.632
```

This is a better route signal than standalone `cattn`: it keeps full prefill
hidden/KV gates passing and improves the integrated prefill from `1135.396 ms`
to `979.765 ms`, or `1.159x` over the `ln2_mlp` hybrid baseline. It is also
`1.082x` faster than the previous `ln1 ORT + cattn RKNN` composition. The
remaining bottleneck is now clearer: the attention suffix ORT cost is still
about `486 ms`, so the next high-value experiment should target a stable,
coarser attention suffix/projection boundary or a lower-overhead runner. This
still is not production proof until service TTFA, chunk gaps, and ASR roundtrip
quality pass with the same artifacts.

The route is also wired into the streaming backend as the experimental
`MOSS_ORT_HYBRID_SPLIT=ln1_cattn` path. RK3576 strict backend smoke with the
combined route bundle completed two streaming chunks:

```text
MOSS_ORT_HYBRID_SPLIT=ln1_cattn
MOSS_ORT_HYBRID_STRICT=1
MOSS_ORT_HYBRID_DIR=/home/cat/moss-rknn-hybrid-baseline
MOSS_ORT_HYBRID_RKNN_DIR=/home/cat/moss-rknn-ln1-cattn-route

preload_ms=18572.624
chunks=2, audio_frames=2, wall_ms=1878.357
ttfa_ms=1195
prefill_ms=1007.230
hybrid_prefill_ms=1002.122
first attention_kind=rknn_ln1_cattn_suffix_ort
first mlp_kind=rknn_ln2_mlp for all 12 layers
```

This smoke proves the route is service-addressable and streaming-capable. It
does not yet prove production quality because it only generated two frames and
did not run ASR roundtrip. The next production gate should run the same
`ln1_cattn` bundle through the existing service profiler with a realistic
`max_new_frames` target, then roundtrip ASR if TTFA/chunk-count pass.

The 20-frame service profile has now passed the streaming gate with the same
Junhao/314 production voice settings as the canonical ORT fallback, strict
hybrid mode, and all 12 `ln1_cattn` RKNN islands:

```text
evidence=docs/evidence/moss/rk3576-moss-service-profile-ln1-cattn-production.json
MOSS_ORT_HYBRID_SPLIT=ln1_cattn
MOSS_ORT_HYBRID_STRICT=1
MOSS_ORT_HYBRID_LAYERS=all

/tts/stream first_payload=1108.090 ms, wall=4635.766 ms
/dialogue first_payload=1079.501 ms, wall=4677.058 ms
dialogue binary_chunks=9, max_payload_gap=577.487 ms
manifest_validated=true, voice=Junhao, seed=314, codec_batch_frames=3
gates.passed=true
```

This is the strongest production-shaped RKNN route so far because it validates
real FastAPI streaming rather than only isolated prefill parity. It is not yet a
promotion candidate: the canonical full-ORT production profile on the same
Junhao/314 setup still measured `/tts/stream first_payload=941.344 ms` and
`/dialogue wall=4642.550 ms`. The practical conclusion is that `ln1_cattn`
reduces long-prompt prefill cost, but the current service path pays enough
RKNN/ORT handoff and fixed sampler/codec cost that it does not beat the tuned
short-prompt ORT fallback. The next high-performance route should therefore
target one of two things before more service integration: reduce handoff and
remaining attention suffix overhead inside prefill, or move a numerically stable
sampler/decode subpath to NPU so the first-audio path improves beyond prefill
alone.

Backend-stage profiling of the same Junhao/314, 20-frame production-shaped
request confirms why this route is not a service win:

```text
full ORT evidence=docs/evidence/moss/rk3576-moss-backend-stage-ort-junhao314-frames20.json
  prefill=933.188 ms
  ttfa=1111 ms
  wall=4786.208 ms
  first sampler=87.991 ms, first codec=78.142 ms
  max decode=69.609 ms, max sampler=87.991 ms, max codec=145.357 ms

ln1_cattn evidence=docs/evidence/moss/rk3576-moss-backend-stage-ln1-cattn-junhao314-frames20.json
  prefill=1013.763 ms
  hybrid_prefill=1010.426 ms
  ttfa=1219 ms
  wall=5157.021 ms
  first sampler=101.373 ms, first codec=94.404 ms
  max decode=75.302 ms, max sampler=101.373 ms, max codec=135.219 ms
  layer attention suffix total=719.963 ms
  RKNN MLP total=199.272 ms
```

The important detail is the ratio inside hybrid prefill: the RKNN MLP portion is
only about `199 ms`, while the remaining ORT attention suffix is about `720 ms`
and every layer still crosses the ORT/RKNN boundary. For short production
prompts, this coarse hybrid path adds about `81 ms` to prefill, `108 ms` to
TTFA, and `371 ms` to 20-frame wall time versus full ORT. It remains useful for
long-prompt diagnostics, but the production optimization focus should move away
from coarse prefill islands unless the runner can eliminate most per-layer
handoff or replace the attention suffix. Otherwise the next meaningful NPU
target is a sampler/decode/codec path that reduces the fixed first-audio cost.

## Remaining Work

- Send or mirror the RKLLM upstream reproducer generated by
  `package_moss_rkllm_reproducer.py` and wait for an official answer on whether
  GPT2-style LayerNorm-with-bias custom blocks are supported. Do not spend more
  local time tuning RKLLM input flags unless official guidance changes the model
  structure or custom config.
- Investigate a sampler-aware RKNN path before any production promotion. The current RKNN FP16 prefill MLP path fails sampler token parity even when hidden cosine is high; `bf16`/`tf32`/`optimization_level=0` are not usable fixes, and even the safer `fc_out_only` split only has short-probe parity for selected combinations. The sampler-side `sampler_mlps0` island is accurate and faster than ORT, but the all-at-once text-head + 17-MLP suffix split failed token parity because it violates local-block sequential dependencies. Candidate directions: a true per-local-block sampler runner, sampler-logit correction on CPU, deterministic/local-code constrained sampling that preserves audio quality, or a different graph partition where the final sampler input remains full-ORT-equivalent.
- Continue prefill surgery only if it removes most attention suffix or per-layer handoff overhead. The all-layer `ln1_cattn` integration is accurate and faster than both `ln2_mlp` baseline and the narrower `cattn` route in the isolated long-prompt verifier, reaching `979.765 ms` prefill (`1.159x` over baseline), but backend-stage Junhao/314 evidence shows it is slower than full ORT for the short production prompt (`prefill 1013.763 ms` vs `933.188 ms`). Service promotion still requires a real TTFA/wall-time win and ASR roundtrip quality, not just hidden/KV parity.
- Fix or replace the experimental voice-prefix KV cache path. Current TTFA improves but quality fails, so it remains opt-in only.
- Continue codec split implementation from the verified boundaries: front RKNN
  through Q/K/V, CPU RoPE/attention, suffix RKNN for out-projection + FFN. The
  layer-0 suffix island is verified on RK3576; the next step is to generate the
  same suffix pattern for all 12 codec layers and run parity/latency as a
  streaming pipeline.
- Split sampler/decode further before RKNN conversion. The current monolithic sampler/decode graphs load but crash on RK3576 inference.
- Convert fixed-shape RKNN buckets and generate `moss-rknn-manifest.json` only after single-bucket RK3576 inference is stable.
- Implement and compile `/opt/rkvoice-workers/moss_rknn_worker` against RKNN C API.
- Keep full ORT fallback as the default until hybrid ASR roundtrip passes with manifest-verified artifacts.
