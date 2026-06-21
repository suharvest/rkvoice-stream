#!/usr/bin/env python3
"""WSL2 fixed-path runner for the RKLLM hidden semantic variant probe."""

from __future__ import annotations

from pathlib import Path

import compare_moss_hf_variants_to_rkllm_hidden as probe


if __name__ == "__main__":
    import sys

    print("variant wrapper start", flush=True)
    print(f"probe module: {probe.__file__}", flush=True)
    sys.argv = [
        "compare_moss_hf_variants_to_rkllm_hidden.py",
        "--model-dir",
        "/home/harve/models/moss-rkllm-embed-only-folded-scaffold",
        "--dump",
        "/home/harve/project/rkvoice-stream/docs/evidence/moss/rk3576-moss-rkllm-hidden-vs-onnx-s8-dump.npz",
        "--assets",
        "/home/harve/models/moss-rkllm-embed-only-folded-scaffold/moss_rkllm_runtime_assets.npz",
        "--json-out",
        "/home/harve/project/rkvoice-stream/docs/evidence/moss/wsl2-moss-hf-variants-vs-rkllm-s8.json",
    ]
    Path(sys.argv[-1]).parent.mkdir(parents=True, exist_ok=True)
    print(f"argv: {sys.argv}", flush=True)
    print("calling probe.main", flush=True)
    try:
        ret = probe.main()
    except BaseException as exc:
        print(f"probe.main raised {type(exc).__name__}: {exc!r}", flush=True)
        raise
    print(f"probe.main returned {ret}", flush=True)
    raise SystemExit(ret)
