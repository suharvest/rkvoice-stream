"""Pytest path setup for the archived MOSS NPU R&D harness.

These tests exercise deep R&D scaffolding (RKLLM exploration, RKNN codec
islands, graph inspection, parity probes) that used to live in the production
tree. After the move to ``experiments/moss/`` the scripts under
``experiments/moss/scripts/`` are imported as top-level modules, so that
directory must be on ``sys.path``. This mirrors how the production tree relied
on the repo root being importable for ``models.tts.moss.*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "scripts"
# Repo root (third_party/rkvoice-stream) so the few imports of *production*
# helpers that stayed behind still resolve, e.g.
# ``from models.tts.moss.convert_moss_rknn import sha256_file``.
REPO_ROOT = HERE.parents[1]

for p in (SCRIPTS_DIR, REPO_ROOT):
    if p.is_dir():
        sys.path.insert(0, str(p))
