#!/usr/bin/env python3
"""List quantizable ONNX node names grouped by Kokoro generator area."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx


def _group(name: str) -> str:
    if "m_source" in name:
        return "source"
    if "noise_convs" in name or "noise_res" in name:
        return "noise"
    if "ups." in name:
        return "ups"
    if "resblocks.0" in name or "resblocks.1" in name or "resblocks.2" in name:
        return "resblocks_0_2"
    if "resblocks.3" in name or "resblocks.4" in name or "resblocks.5" in name:
        return "resblocks_3_5"
    if "conv_post" in name:
        return "post"
    if "adain" in name and "/fc/" in name:
        return "adain_fc"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ops", nargs="*", default=["Conv", "ConvTranspose", "Gemm", "MatMul"])
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    groups: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for node in model.graph.node:
        if node.op_type not in args.ops:
            continue
        groups[_group(node.name)].append({"name": node.name, "op": node.op_type})

    result = {
        group: {"count": len(items), "ops": dict(collections.Counter(item["op"] for item in items)), "nodes": items}
        for group, items in sorted(groups.items())
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
