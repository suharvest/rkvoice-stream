"""Online Kokoro v1.0 prefix adapter for the Conv-only RKNN tail.

The adapter deliberately stops at the ``j1`` boundary.  It reuses the
validated long32 runtime contexts for main0/noise0/rb0..2/main1/noise1 and
derives the eighteen AdaIN FiLM vectors from the rb3/rb4/rb5 style FC
parameters.  The Conv-only tail can then consume the returned arrays without
loading the legacy sinusoidal branch implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future
import threading
import time
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PrefixBoundary:
    """Inputs required by the Conv-only tail runner."""

    j1: np.ndarray
    gamma: np.ndarray  # [1, 18, 128]
    beta: np.ndarray   # [1, 18, 128]
    timings: Mapping[str, float] = field(default_factory=dict)


class OnlinePrefixAdapter:
    """Run the official prefix through an already-preloaded long32 runtime.

    ``runtime`` is intentionally injected so this module does not duplicate
    model loading, manifest validation, or device selection.  It must be a
    ``KokoroLong32Runtime`` loaded with the same dual-provider bundle as the
    approved tail fixtures.
    """

    def __init__(self, runtime: Any):
        required = ("ready", "contexts", "branches", "noise", "_noise0_for", "pool", "_lifecycle_lock")
        if any(not hasattr(runtime, name) for name in required):
            raise TypeError("runtime is not a loaded KokoroLong32Runtime")
        if not runtime.ready:
            raise RuntimeError("runtime.preload() is required")
        self.runtime = runtime

    @staticmethod
    def _finite(value: Any, name: str) -> np.ndarray:
        out = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        if not np.isfinite(out).all():
            raise ValueError(f"{name} contains non-finite values")
        return out

    def _film(self, style: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract rb3/rb4/rb5 AdaIN vectors in tail site order."""
        s = self._finite(style, "s")
        if s.shape != (1, 128):
            raise ValueError(f"s shape must be [1,128], got {s.shape}")
        gammas: list[np.ndarray] = []
        betas: list[np.ndarray] = []
        # Runtime branch order is block-major; within each rb block the
        # exported FC params are half-major, while the tail consumes
        # residual-stage-major sites.
        for block in (3, 4, 5):
            branch = self.runtime.branches[block]
            if len(branch.params) != 6:
                raise ValueError(f"rb{block} must expose six AdaIN parameter sets")
            for stage in range(3):
                for half in range(2):
                    index = half * 3 + stage
                    weight, bias, _alpha = branch.params[index]
                    # _Branch uses style = s @ w.T + b; first 128 are gamma,
                    # second 128 are beta.  Alpha is intentionally not exported.
                    w = self._finite(weight, f"rb{block} FC weight")
                    b = self._finite(bias, f"rb{block} FC bias")
                    style_out = s @ w.T + b
                    gammas.append(style_out[:, :128])
                    betas.append(style_out[:, 128:])
        return (np.ascontiguousarray(np.stack(gammas, axis=1), dtype=np.float32),
                np.ascontiguousarray(np.stack(betas, axis=1), dtype=np.float32))

    def run_prefix(self, tensors: Mapping[str, Any]) -> PrefixBoundary:
        """Return ``j1``, gamma and beta for one frontend boundary tensor set."""
        x = self._finite(tensors["x"], "x")
        s = self._finite(tensors["s"], "s")
        t = int(x.shape[-1]) if x.ndim == 3 else -1
        if x.shape != (1, 512, t) or t <= 0:
            raise ValueError(f"x shape must be [1,512,T], got {x.shape}")
        route_ts = getattr(self.runtime, "route_ts", None)
        if route_ts is None or t not in route_ts:
            raise ValueError(f"T={t} is not an approved runtime route; route_ts={route_ts}")
        f = 60 * t + 1
        har = self._finite(tensors["har"], "har")
        if har.shape != (1, 22, f):
            raise ValueError(f"har shape must be [1,22,{f}], got {har.shape}")
        pool = self.runtime.pool
        if pool is None: raise RuntimeError("runtime worker pool is unavailable")
        timings: dict[str, float] = {}
        futures: list[Future] = []
        wall = time.perf_counter()
        def submit(name, fn, *args):
            def timed():
                started = time.perf_counter()
                try: return fn(*args)
                finally: timings[name] = (time.perf_counter() - started) * 1000.0
            future = pool.submit(timed); futures.append(future); return future
        def drain():
            for future in futures:
                future.cancel()
            for future in futures:
                try: future.result()
                except BaseException: pass
        # Hold the runtime lifecycle gate until every submitted worker has drained.
        with self.runtime._lifecycle_lock:
            try:
                c = self.runtime.contexts
                f_main0 = submit("main0", c["main0"].run, [x])
                f_noise0 = submit("noise0", self.runtime._noise0_for(t).run, [s, har])
                f_noise1 = submit("noise1", self.runtime.noise.run, har, s)
                f_film = submit("film", self._film, s)
                main0, noise0 = f_main0.result(), f_noise0.result()
                j0 = np.asarray(main0 + noise0, dtype=np.float32)
                rb_futures = [submit(f"rb{i}", c[f"rb{i}"].run, [j0, s]) for i in range(3)]
                rb = [future.result() for future in rb_futures]
                f_main1 = submit("main1", c["main1"].run, [np.asarray((rb[0] + rb[1] + rb[2]) / np.float32(3), np.float32)])
                main1 = f_main1.result()
                noise1 = f_noise1.result()
                gamma, beta = f_film.result()
                j1 = np.ascontiguousarray(main1 + noise1, dtype=np.float32)
                critical = max(timings.get("main0", 0), timings.get("noise0", 0)) + max(timings.get(f"rb{i}", 0) for i in range(3)) + timings.get("main1", 0)
                critical = max(critical, max(timings.get("noise1", 0), timings.get("film", 0)))
                timings["prefix_wall_ms"] = (time.perf_counter() - wall) * 1000.0
                timings["prefix_critical_path_ms"] = critical
                return PrefixBoundary(j1=j1, gamma=gamma, beta=beta, timings=timings)
            finally:
                drain()

    def run_frontend(self, frontend: Any, text: str, *, language: str,
                     speed: float = 1.0, target_d: int | None = None) -> PrefixBoundary:
        """Prepare/render official multilingual text, then run the prefix."""
        if target_d is None:
            raise ValueError("target_d is required; pass the approved profile snap explicitly")
        prepared = frontend.prepare(text, language=language, speed=speed)
        rendered = frontend.render(prepared, target_D=target_d)
        return self.run_prefix(rendered)


def boundary_metrics(actual: PrefixBoundary, expected: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Compare generated boundary arrays against an approved cache."""
    out: dict[str, dict[str, float]] = {}
    for name, got in (("j1", actual.j1), ("gamma", actual.gamma), ("beta", actual.beta)):
        ref = np.asarray(expected[name], dtype=np.float32)
        if got.shape != ref.shape:
            raise ValueError(f"{name} shape mismatch: {got.shape} != {ref.shape}")
        delta = got.astype(np.float64) - ref.astype(np.float64)
        denom = max(float(np.linalg.norm(ref.ravel())), 1e-12)
        gnorm = max(float(np.linalg.norm(got.ravel())), 1e-12)
        rnorm = max(float(np.linalg.norm(ref.ravel())), 1e-12)
        out[name] = {
            "maxabs": float(np.max(np.abs(delta))),
            "rel_l2": float(np.linalg.norm(delta) / denom),
            "cosine": float(np.dot(got.ravel(), ref.ravel()) / (gnorm * rnorm)),
        }
    return out
