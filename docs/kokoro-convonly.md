# Kokoro ConvOnly

Kokoro ConvOnly is a first-class RKVoice Stream TTS backend. Select it with
`backend: kokoro_convonly`; OpenVoiceStream can use the same library directly
through its normal RK TTS adapter.

The runtime supports English, Chinese, and Japanese. Frontend dependencies are
optional extras (`kokoro-en`, `kokoro-zh`, `kokoro-ja`); the Japanese UniDic
Lite dictionary remains an explicit opt-in dependency. RKNN model files,
frontend data, and the Japanese dictionary are external read-only mounts.
The published Hugging Face repository is
`harvestsu/seeed-local-voice-rk-artifacts`, with platform paths
`rk3576/kokoro-convonly-v1_0/` and `rk3588/kokoro-convonly-v1_0/`.
The immutable release revision is
`3f8d58c8446ec4b18891624ad4ae4ce75e0f3d3e`.
The locally qualified manifest SHA256 values are `24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9`
for RK3576 and `83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e`
for RK3588.

```yaml
tts:
  backend: kokoro_convonly
  bundle_root: /opt/kokoro-convonly/rk3576
  platform: rk3576
  manifest_sha256: 24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9
  intra: 6
  inter: 1
```

The equivalent environment is `TTS_BACKEND=kokoro_convonly`,
`KOKORO_CONVONLY_ROOT=/opt/kokoro-convonly/rk3576`,
`RK_PLATFORM=rk3576`, and
`KOKORO_CONVONLY_MANIFEST_SHA256=24244b7054bc3626fc22f4ee9bc013ef63aaa5cf409675cafbc10e1c53957ed9`.
The manifest is a `kokoro.convonly.bundle.v1` receipt and pins every staged
file. Generate or check it with:

```bash
python tools/package_kokoro_convonly_bundle.py \
  /staging/rk3576/kokoro-convonly-v1_0 --platform rk3576

python tools/package_kokoro_convonly_bundle.py \
  /opt/kokoro-convonly/rk3576 --platform rk3576 --check
```

RK3576 uses the native ConvOnly tail and its approved profile set; RK3588 uses
the Python tail and its platform profile set. Both platforms use sentence
streaming, safe request cancellation, and release-before-switch resource
ownership. The CPU generator is a profile fallback for duration/profile cases
when explicitly selected and present in the manifest. An NPU preload failure
does not silently switch the service to CPU-only mode.

The native tail requires the external RKNN SDK headers and `librknnrt`; the
SDK is not downloaded or bundled by this repository. Build it with
`make -C native/kokoro_convonly RKNN_INC=/path/to/include
RKNN_LIB=/path/to/lib`.

## Release checklist

Build the wheel from a clean checkout, install the needed extras, and prepare
the two platform bundles for the Hugging Face release target above. The Docker image remains one AK unified image;
only the read-only bundle mount and platform profile select the board
resources. Registry image digests and HF commit IDs are release metadata.
