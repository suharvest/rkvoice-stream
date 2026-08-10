"""Platform-aware RKNNLite runtime initialisation.

Rockchip NPUs split into two families with an incompatible ``init_runtime()``
calling convention:

* **Multi-core** parts (rk3576 = 2 cores, rk3588 = 3 cores) expose a
  *core mask* — ``init_runtime(core_mask=RKNNLite.NPU_CORE_0)`` — so a caller
  can pin a model to a specific NPU core (e.g. ASR encoder on CORE_1, TTS
  vocoder on CORE_0).

* **Single-core** SoCs (e.g. **rv1126b**, one NPU core) have **no core-mask
  concept**. They MUST be initialised with a plain ``init_runtime()``; passing
  any ``core_mask=`` (even ``NPU_CORE_0``) makes the runtime error out.

The PCIe coprocessor rk1828 is also ``npu_cores == 1`` but is addressed by its
``device_id`` (PCIe BDF), not a core mask — it is flagged ``is_coprocessor`` and
is treated as *not* single-core here (its own loader passes ``device_id``).

``init_runtime_for_platform`` centralises this decision so every RK backend
(sensevoice / paraformer / qwen3 encoder / …) initialises consistently and
rv1126b works across all of them.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from . import get_platform


def platform_is_single_core(platform: Optional[str] = None) -> bool:
    """Return True for a single-core host NPU (no core-mask concept).

    ``platform`` defaults to the ``RK_PLATFORM`` env var. Unknown / unset
    platforms return False (preserve the historical multi-core default).
    A PCIe coprocessor profile (``is_coprocessor``) returns False — it is
    addressed by ``device_id``, not by the maskless path.
    """
    if not platform:
        platform = os.environ.get("RK_PLATFORM", "")
    if not platform:
        return False
    try:
        prof = get_platform(platform)
    except Exception:
        return False
    return prof.npu_cores == 1 and not prof.is_coprocessor


def init_runtime_for_platform(
    rknn,
    *,
    platform: Optional[str] = None,
    core_mask: Union[str, int, None] = "NPU_CORE_0",
    force_single_core: bool = False,
) -> int:
    """Call ``rknn.init_runtime()`` the right way for ``platform``.

    On a single-core platform (or ``force_single_core=True``) calls
    ``init_runtime()`` with no mask. On multi-core parts resolves ``core_mask``
    against ``RKNNLite`` (string name such as ``"NPU_CORE_0"``, or a raw int)
    and passes it through. Returns the ``init_runtime()`` int status so callers
    keep their own error messages.

    ``platform`` defaults to the ``RK_PLATFORM`` env var, so most callers can
    omit it.
    """
    from rknnlite.api import RKNNLite

    if force_single_core or platform_is_single_core(platform):
        return rknn.init_runtime()

    if isinstance(core_mask, str):
        core = getattr(RKNNLite, core_mask, RKNNLite.NPU_CORE_AUTO)
    elif core_mask is None:
        core = RKNNLite.NPU_CORE_AUTO
    else:
        core = core_mask
    return rknn.init_runtime(core_mask=core)
