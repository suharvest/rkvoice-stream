# MOSS NPU R&D archive

Archived deep R&D scaffolding for the **MOSS-TTS-Nano** backend — RKLLM exploration,
RKNN codec/sampler islands, graph inspection, and parity probes. These are *not* part of
the production path (ORT is the supported CPU fallback; hybrid is experimental-supported).
Conclusions: see [`../../docs/moss-rknn-rk3576.md`](../../docs/moss-rknn-rk3576.md).

- `scripts/` — R&D scripts (RKLLM/RKNN island export, probes, parity verifiers)
- `evidence/` — captured run evidence (JSON / logs) that the tests below pin
- `tests/` — evidence-lock and unit tests for the above

Run: `python3 -m pytest experiments/moss/tests/`
