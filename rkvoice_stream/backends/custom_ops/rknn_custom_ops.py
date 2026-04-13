"""
RKNN Custom Operator Registration via ctypes.

The rknnlite Python API does not expose rknn_register_custom_ops(),
so we call it directly via ctypes on librknnrt.so.

Usage:
    from rknnlite.api import RKNNLite
    from rkvoice_stream.backends.custom_ops.rknn_custom_ops import register_cst_sin

    r = RKNNLite()
    r.load_rknn("model.rknn")
    r.init_runtime()
    refs = register_cst_sin(r)  # keep refs alive until release!
    # ... r.inference(...) now works with CstSin ops ...
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---- Constants from rknn_api.h / rknn_custom_op.h ----

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_TARGET_TYPE_CPU = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1

# ---- ctypes struct definitions (must match ABI exactly) ----


class _RknnGpuOpContext(ctypes.Structure):
    _fields_ = [
        ("cl_context", ctypes.c_void_p),
        ("cl_command_queue", ctypes.c_void_p),
        ("cl_kernel", ctypes.c_void_p),
    ]


class _RknnCustomOpContext(ctypes.Structure):
    _fields_ = [
        ("target", ctypes.c_int),
        ("internal_ctx", ctypes.c_uint64),  # aarch64
        ("gpu_ctx", _RknnGpuOpContext),
        ("priv_data", ctypes.c_void_p),
    ]


class _RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class _RknnTensorMem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class _RknnCustomOpTensor(ctypes.Structure):
    _fields_ = [
        ("attr", _RknnTensorAttr),
        ("mem", _RknnTensorMem),
    ]


_COMPUTE_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(_RknnCustomOpContext),
    ctypes.POINTER(_RknnCustomOpTensor), ctypes.c_uint32,
    ctypes.POINTER(_RknnCustomOpTensor), ctypes.c_uint32,
)
_DESTROY_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_RknnCustomOpContext))


class _RknnCustomOp(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("target", ctypes.c_int),
        ("op_type", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("cl_kernel_name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("cl_kernel_source", ctypes.c_char_p),
        ("cl_source_size", ctypes.c_uint64),
        ("cl_build_options", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("init", _COMPUTE_FUNC),
        ("prepare", _COMPUTE_FUNC),
        ("compute", _COMPUTE_FUNC),
        ("compute_native", _COMPUTE_FUNC),
        ("destroy", _DESTROY_FUNC),
    ]


def register_custom_ops(
    rknn_lite_obj,
    lib_path: str = "/opt/tts/lib/libcstops.so",
    librknnrt_path: str = "/usr/lib/librknnrt.so",
) -> Optional[tuple]:
    """
    Register all custom CPU operators (CstSin, CstMul, CstPow, CstAdd,
    CstInstanceNorm, PiperSplineCoupling) with an initialized RKNNLite context.

    Must be called AFTER load_rknn() + init_runtime().

    Returns:
        Tuple of ctypes references that must be kept alive (prevents GC).
        Returns None on failure.
    """
    try:
        rt = rknn_lite_obj.rknn_runtime
        if rt is None:
            log.error("custom_ops: rknn_runtime is None")
            return None

        ctx_value = rt.context
        log.info("custom_ops: rknn_context = 0x%x", ctx_value)

        lib = ctypes.CDLL(lib_path)

        # Map op_type -> C function name in libcstops.so
        op_defs = [
            (b"CstSin",          "cst_sin_compute"),
            (b"CstMul",          "cst_mul_compute"),
            (b"CstPow",          "cst_pow_compute"),
            (b"CstAdd",          "cst_add_compute"),
            (b"CstInstanceNorm", "cst_instance_norm_compute"),
            (b"PiperSplineCoupling", "cst_spline_coupling_compute"),
        ]

        # Build list of ops that actually exist in the shared library
        valid_ops = []
        callbacks = []  # prevent GC
        for (op_type, func_name) in op_defs:
            fn = getattr(lib, func_name, None)
            if fn is None:
                log.warning("custom_ops: %s not found in %s, skipping", func_name, lib_path)
                continue
            cb = _COMPUTE_FUNC(fn)
            callbacks.append(cb)
            valid_ops.append((op_type, cb))

        if not valid_ops:
            log.error("custom_ops: no valid ops found in %s", lib_path)
            return None

        n_ops = len(valid_ops)
        ops = (_RknnCustomOp * n_ops)()
        for i, (op_type, cb) in enumerate(valid_ops):
            ctypes.memset(ctypes.byref(ops[i]), 0, ctypes.sizeof(_RknnCustomOp))
            ops[i].version = 1
            ops[i].target = RKNN_TARGET_TYPE_CPU
            ops[i].op_type = op_type
            ops[i].compute = cb

        librknnrt = ctypes.CDLL(librknnrt_path)
        register_fn = librknnrt.rknn_register_custom_ops
        register_fn.restype = ctypes.c_int
        register_fn.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(_RknnCustomOp),
            ctypes.c_uint32,
        ]

        ret = register_fn(ctx_value, ops, n_ops)
        if ret != 0:
            log.error("custom_ops: rknn_register_custom_ops failed (ret=%d)", ret)
            return None

        log.info("custom_ops: %d ops registered successfully", n_ops)
        return (lib, callbacks, ops, librknnrt)

    except Exception:
        log.exception("custom_ops: registration failed")
        return None
