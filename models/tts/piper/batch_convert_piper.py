#!/usr/bin/env python3
"""Batch download and convert Piper TTS models to RKNN.

Downloads Piper voice models from HuggingFace and converts them for hybrid
NPU deployment: encoder.onnx (CPU ORT, dynamic shapes) + flow_decoder.rknn
(RKNN NPU, fixed mel_len=256). RTF ~0.07 on RK3576.

Usage:
  python batch_convert_piper.py --target rk3576 --output-dir /tmp/piper-models
  python batch_convert_piper.py --target rk3588 --languages en_US,zh_CN
  python batch_convert_piper.py --target rk3576 --languages all
"""

import os
import sys
import json
import argparse
import tempfile
import urllib.request
import urllib.error
import time
import traceback

# ---------------------------------------------------------------------------
# Model registry: language code → (hf_path, model_name)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- Existing / already deployed ---
    'en_US':  ('en/en_US/lessac/medium',      'en_US-lessac-medium'),
    'zh_CN':  ('zh/zh_CN/huayan/medium',      'zh_CN-huayan-medium'),
    'ja_JP':  ('ja/ja_JP/kokoro/medium',      'ja_JP-kokoro-medium'),
    'de_DE':  ('de/de_DE/thorsten/medium',    'de_DE-thorsten-medium'),
    'fr_FR':  ('fr/fr_FR/siwis/medium',       'fr_FR-siwis-medium'),

    # --- Priority expansion languages ---
    'es_ES':  ('es/es_ES/davefx/medium',      'es_ES-davefx-medium'),
    'es_MX':  ('es/es_MX/claude/high',        'es_MX-claude-high'),
    'it_IT':  ('it/it_IT/riccardo/x_low',     'it_IT-riccardo-x_low'),
    'ru_RU':  ('ru/ru_RU/irina/medium',       'ru_RU-irina-medium'),
    'pt_BR':  ('pt/pt_BR/faber/medium',       'pt_BR-faber-medium'),
    'nl_NL':  ('nl/nl_NL/mls_5809/low',       'nl_NL-mls_5809-low'),
    'pl_PL':  ('pl/pl_PL/darkman/medium',     'pl_PL-darkman-medium'),
    'ar_JO':  ('ar/ar_JO/kareem/medium',      'ar_JO-kareem-medium'),
    'tr_TR':  ('tr/tr_TR/dfki/medium',        'tr_TR-dfki-medium'),
    'vi_VN':  ('vi/vi_VN/vivos/x_low',        'vi_VN-vivos-x_low'),
    # ko_KR: not available in Piper — skipped intentionally
    'uk_UA':  ('uk/uk_UA/lada/x_low',         'uk_UA-lada-x_low'),
    'sv_SE':  ('sv/sv_SE/nst/medium',         'sv_SE-nst-medium'),
    'cs_CZ':  ('cs/cs_CZ/jirka/medium',       'cs_CZ-jirka-medium'),
}

HF_BASE = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0'


def download_file(url: str, dest: str, max_retries: int = 3) -> bool:
    """Download a file from URL to dest path. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Downloading: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            with open(dest, 'wb') as f:
                f.write(data)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"    Saved: {dest} ({size_mb:.1f} MB)")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    ERROR: 404 Not Found — {url}")
                return False
            print(f"    HTTP error {e.code} (attempt {attempt}/{max_retries}): {e}")
        except Exception as e:
            print(f"    Download error (attempt {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            time.sleep(5 * attempt)
    return False


def build_rknn(fixed_onnx: str, rknn_path: str, target: str) -> bool:
    """Run RKNN build and export. Returns True on success."""
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    ret = rknn.config(target_platform=target, optimization_level=0)
    if ret != 0:
        print(f"    RKNN config failed: {ret}")
        return False

    ret = rknn.load_onnx(model=fixed_onnx)
    if ret != 0:
        print(f"    RKNN load_onnx failed: {ret}")
        return False

    print(f"    RKNN build (target={target})...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"    RKNN build failed: {ret}")
        return False

    ret = rknn.export_rknn(rknn_path)
    rknn.release()
    if ret != 0:
        print(f"    RKNN export failed: {ret}")
        return False

    sz = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"    RKNN exported: {rknn_path} ({sz:.1f} MB)")
    return True


def convert_language_hybrid(lang: str, target: str, output_dir: str,
                            mel_len: int = 256,
                            keep_intermediate: bool = False) -> dict:
    """Download, split, and convert a single language model (hybrid mode).

    Hybrid: encoder.onnx (CPU ORT) + flow_decoder.rknn (NPU).
    Returns a result dict with keys: lang, status, error, encoder_path, rknn_path, config_path.
    """
    result = {'lang': lang, 'status': 'fail', 'encoder_path': None,
              'rknn_path': None, 'config_path': None, 'error': None}

    if lang not in MODEL_REGISTRY:
        result['error'] = f"Unknown language: {lang}"
        return result

    hf_subpath, model_name = MODEL_REGISTRY[lang]
    onnx_url   = f"{HF_BASE}/{hf_subpath}/{model_name}.onnx"
    config_url = f"{HF_BASE}/{hf_subpath}/{model_name}.onnx.json"

    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    raw_onnx    = os.path.join(lang_dir, f"{model_name}.onnx")
    config_src  = os.path.join(lang_dir, f"{model_name}.onnx.json")
    config_dst  = os.path.join(lang_dir, "config.json")

    # --- Download ONNX ---
    print(f"\n  [{lang}] Downloading ONNX model...")
    if not download_file(onnx_url, raw_onnx):
        result['error'] = f"Download failed: {onnx_url}"
        return result

    # --- Download config JSON ---
    print(f"  [{lang}] Downloading config JSON...")
    if not download_file(config_url, config_src):
        print(f"  [{lang}] WARNING: config JSON download failed (non-fatal)")
    else:
        # Copy to config.json (canonical name) and keep original
        import shutil
        shutil.copy2(config_src, config_dst)
        # Also keep as model.onnx.json for backward compat
        model_onnx_json = os.path.join(lang_dir, "model.onnx.json")
        if not os.path.exists(model_onnx_json):
            shutil.copy2(config_src, model_onnx_json)
        result['config_path'] = config_dst

    # --- Split into encoder + flow_decoder ---
    print(f"  [{lang}] Splitting into encoder + flow_decoder...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from split_piper_vits import split_model
        encoder_path, fd_onnx_path = split_model(
            raw_onnx, lang_dir, mel_len=mel_len,
        )
        result['encoder_path'] = encoder_path
    except Exception as e:
        result['error'] = f"Split failed: {e}\n{traceback.format_exc()}"
        return result

    # --- Build RKNN from flow_decoder.onnx ---
    rknn_path = os.path.join(lang_dir, "flow_decoder.rknn")
    print(f"  [{lang}] Building RKNN from flow_decoder ({target})...")
    try:
        ok = build_rknn(fd_onnx_path, rknn_path, target)
        if not ok:
            result['error'] = "RKNN build returned False"
            return result
    except ImportError:
        print(f"  [{lang}] WARNING: rknn.api not available, skipping RKNN build.")
        result['status'] = 'onnx_only'
        result['rknn_path'] = fd_onnx_path
        return result
    except Exception as e:
        result['error'] = f"RKNN build exception: {e}\n{traceback.format_exc()}"
        return result

    # --- Cleanup intermediate files ---
    if not keep_intermediate:
        for f in [raw_onnx, config_src]:
            try:
                if os.path.exists(f) and f != config_dst:
                    os.remove(f)
            except OSError:
                pass
        # Keep flow_decoder.onnx only if requested
        try:
            if os.path.exists(fd_onnx_path):
                os.remove(fd_onnx_path)
        except OSError:
            pass

    result['status'] = 'success'
    result['rknn_path'] = rknn_path
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch download and convert Piper TTS models to RKNN'
    )
    parser.add_argument(
        '--target', default='rk3588',
        choices=['rk3576', 'rk3588', 'rk3562', 'rv1103', 'rv1106'],
        help='RKNN target platform (default: rk3588)'
    )
    parser.add_argument(
        '--output-dir', default='/tmp/piper-rknn-models',
        help='Directory to write converted models (default: /tmp/piper-rknn-models)'
    )
    parser.add_argument(
        '--languages', default='all',
        help='Comma-separated language codes, or "all" (default: all). '
             f'Available: {", ".join(sorted(MODEL_REGISTRY.keys()))}'
    )
    parser.add_argument(
        '--mel-len', type=int, default=256,
        help='Fixed mel length for hybrid mode flow_decoder (default: 256)'
    )
    parser.add_argument(
        '--keep-intermediate', action='store_true',
        help='Keep raw and fixed ONNX files after RKNN conversion'
    )
    args = parser.parse_args()

    # Resolve language list
    if args.languages.strip().lower() == 'all':
        langs = sorted(MODEL_REGISTRY.keys())
    else:
        langs = [l.strip() for l in args.languages.split(',') if l.strip()]

    print("=" * 60)
    print("Piper TTS -> RKNN Batch Converter")
    print(f"  Target:     {args.target}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Languages:  {', '.join(langs)}")
    print(f"  Mel len:    {args.mel_len}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for lang in langs:
        print(f"\n{'='*60}")
        print(f"Processing: {lang} (mode={args.mode})")
        print('='*60)
        t0 = time.time()

        result = convert_language_hybrid(
            lang=lang,
            target=args.target,
            output_dir=args.output_dir,
            mel_len=args.mel_len,
            keep_intermediate=args.keep_intermediate,
        )

        elapsed = time.time() - t0
        result['elapsed_s'] = round(elapsed, 1)
        results.append(result)

        if result['status'] == 'success':
            rknn_sz = os.path.getsize(result['rknn_path']) / (1024 * 1024) if result['rknn_path'] else 0
            print(f"\n  [{lang}] SUCCESS in {elapsed:.0f}s -- {result['rknn_path']} ({rknn_sz:.1f} MB)")
        elif result['status'] == 'onnx_only':
            print(f"\n  [{lang}] ONNX ONLY (no RKNN toolkit) in {elapsed:.0f}s -- {result['rknn_path']}")
        else:
            print(f"\n  [{lang}] FAILED in {elapsed:.0f}s -- {result['error']}")

    # Save results JSON
    results_file = os.path.join(args.output_dir, 'conversion_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    succeeded = [r for r in results if r['status'] == 'success']
    onnx_only = [r for r in results if r['status'] == 'onnx_only']
    failed    = [r for r in results if r['status'] == 'fail']

    for r in succeeded:
        sz = os.path.getsize(r['rknn_path']) / (1024 * 1024) if r['rknn_path'] else 0
        print(f"  OK   {r['lang']:12s}  {r['elapsed_s']:5.0f}s  {sz:5.1f} MB  {r['rknn_path']}")
    for r in onnx_only:
        print(f"  ONNX {r['lang']:12s}  {r['elapsed_s']:5.0f}s  (no rknn toolkit)  {r['rknn_path']}")
    for r in failed:
        print(f"  FAIL {r['lang']:12s}  {r['elapsed_s']:5.0f}s  {r['error'][:80]}")

    print(f"\nResults: {len(succeeded)} succeeded, {len(onnx_only)} onnx-only, {len(failed)} failed")
    print(f"Results JSON: {results_file}")

    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
