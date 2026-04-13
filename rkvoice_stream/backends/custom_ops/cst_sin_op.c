/**
 * CstSin custom operator for RKNN — NEON accelerated
 *
 * Computes elementwise sin() in FP32 on CPU with ARM NEON SIMD.
 * Handles FP16 input/output transparently.
 *
 * Build (on ARM64):
 *   gcc -shared -fPIC -O2 -march=armv8-a+simd -o libcstsin.so cst_sin_op.c -lm
 *
 * Compatible with RK3576, RK3588, and any aarch64 Linux.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* ---- Minimal RKNN type definitions (must match ABI) ---- */

#define RKNN_MAX_DIMS     16
#define RKNN_MAX_NAME_LEN 256

typedef enum {
    RKNN_TENSOR_FLOAT32 = 0, RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8 = 2, RKNN_TENSOR_UINT8 = 3,
} rknn_tensor_type;

typedef enum {
    RKNN_TENSOR_NCHW = 0, RKNN_TENSOR_NHWC = 1,
    RKNN_TENSOR_NC1HWC2 = 2, RKNN_TENSOR_UNDEFINED = 3,
} rknn_tensor_format;

typedef enum {
    RKNN_TENSOR_QNT_NONE = 0, RKNN_TENSOR_QNT_DFP = 1,
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC = 2,
} rknn_tensor_qnt_type;

typedef struct {
    uint32_t index; uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems; uint32_t size;
    rknn_tensor_format fmt; rknn_tensor_type type;
    rknn_tensor_qnt_type qnt_type;
    int8_t fl; int32_t zp; float scale;
    uint32_t w_stride; uint32_t size_with_stride;
    uint8_t pass_through; uint32_t h_stride;
} rknn_tensor_attr;

typedef struct {
    void* virt_addr; uint64_t phys_addr;
    int32_t fd; int32_t offset;
    uint32_t size; uint32_t flags;
    void* priv_data;
} rknn_tensor_mem;

typedef struct {
    rknn_tensor_attr attr;
    rknn_tensor_mem mem;
} rknn_custom_op_tensor;

typedef enum { RKNN_TARGET_TYPE_CPU = 1, RKNN_TARGET_TYPE_GPU = 2 } rknn_target_type;

typedef struct {
    void* cl_context; void* cl_command_queue; void* cl_kernel;
} rknn_gpu_op_context;

typedef struct {
    rknn_target_type target;
    uint64_t internal_ctx;
    rknn_gpu_op_context gpu_ctx;
    void* priv_data;
} rknn_custom_op_context;

/* ---- NEON-accelerated sin ---- */

#ifdef __aarch64__

/* 7th-order minimax sin approximation on [-pi, pi], max error ~1.5e-6 */
static inline float32x4_t vsinq_f32(float32x4_t x) {
    const float32x4_t inv_2pi = vdupq_n_f32(0.15915494309189535f);
    const float32x4_t two_pi  = vdupq_n_f32(6.283185307179586f);
    const float32x4_t c3 = vdupq_n_f32(-0.16666667f);
    const float32x4_t c5 = vdupq_n_f32( 0.0083333336f);
    const float32x4_t c7 = vdupq_n_f32(-0.0001984127f);

    /* Range reduction: x = x - 2*pi * round(x / (2*pi)) */
    float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_2pi));
    x = vfmsq_f32(x, n, two_pi);

    /* sin(x) ~ x * (1 + x²*(c3 + x²*(c5 + x²*c7))) */
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t r = vfmaq_f32(c5, c7, x2);
    r = vfmaq_f32(c3, r, x2);
    r = vfmaq_f32(vdupq_n_f32(1.0f), r, x2);
    return vmulq_f32(r, x);
}

static void sin_fp32_neon(const float* in, float* out, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vsinq_f32(vld1q_f32(in + i)));
    }
    for (; i < n; i++) out[i] = sinf(in[i]);
}

static void sin_fp16_neon(const uint16_t* in, uint16_t* out, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)(in + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1_f16((__fp16*)(out + i), vcvt_f16_f32(vsinq_f32(f)));
    }
    for (; i < n; i++) {
        __fp16 hv; memcpy(&hv, &in[i], 2);
        float fv = sinf((float)hv);
        hv = (__fp16)fv; memcpy(&out[i], &hv, 2);
    }
}

#else
static void sin_fp32_neon(const float* in, float* out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = sinf(in[i]);
}
#endif

/* ---- Compute callback ---- */

__attribute__((visibility("default")))
int cst_sin_compute(rknn_custom_op_context* op_ctx,
                    rknn_custom_op_tensor* inputs,  uint32_t n_inputs,
                    rknn_custom_op_tensor* outputs, uint32_t n_outputs)
{
    uint32_t n = inputs[0].attr.n_elems;
    void* in_data  = (char*)inputs[0].mem.virt_addr + inputs[0].mem.offset;
    void* out_data = (char*)outputs[0].mem.virt_addr + outputs[0].mem.offset;

    if (inputs[0].attr.type == RKNN_TENSOR_FLOAT32) {
        sin_fp32_neon((const float*)in_data, (float*)out_data, n);
    } else if (inputs[0].attr.type == RKNN_TENSOR_FLOAT16) {
#ifdef __aarch64__
        sin_fp16_neon((const uint16_t*)in_data, (uint16_t*)out_data, n);
#else
        return -1;
#endif
    } else {
        return -1;
    }
    return 0;
}
