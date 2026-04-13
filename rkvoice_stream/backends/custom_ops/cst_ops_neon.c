/**
 * Custom RKNN CPU operators with NEON acceleration.
 * Replaces FP16-problematic ops in Matcha TTS estimator Snake activation
 * and InstanceNorm with FP32 CPU implementations.
 *
 * Ops: CstSin, CstMul, CstPow, CstAdd, CstInstanceNorm
 *
 * Build: gcc -shared -fPIC -O2 -march=armv8-a+simd -o libcstops.so cst_ops_neon.c -lm
 */

#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* ---- Minimal RKNN types (match ABI) ---- */
#define RKNN_MAX_DIMS     16
#define RKNN_MAX_NAME_LEN 256

typedef struct {
    uint32_t index, n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems, size;
    int fmt, type, qnt_type;
    int8_t fl; int32_t zp; float scale;
    uint32_t w_stride, size_with_stride;
    uint8_t pass_through; uint32_t h_stride;
} rknn_tensor_attr;

typedef struct {
    void* virt_addr; uint64_t phys_addr;
    int32_t fd, offset;
    uint32_t size, flags;
    void* priv_data;
} rknn_tensor_mem;

typedef struct {
    rknn_tensor_attr attr;
    rknn_tensor_mem mem;
} rknn_custom_op_tensor;

typedef struct { void *a, *b, *c; } rknn_gpu_op_context;
typedef struct {
    int target; uint64_t internal_ctx;
    rknn_gpu_op_context gpu_ctx; void* priv_data;
} rknn_custom_op_context;

/* ---- Helpers ---- */

static inline void* tensor_data(rknn_custom_op_tensor* t) {
    return (char*)t->mem.virt_addr + t->mem.offset;
}

static inline uint32_t tensor_elems(rknn_custom_op_tensor* t) {
    return t->attr.n_elems;
}

/* FP16 <-> FP32 via NEON */
#ifdef __aarch64__
static inline void fp16_to_fp32_buf(const uint16_t* in, float* out, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)(in + i));
        vst1q_f32(out + i, vcvt_f32_f16(h));
    }
    for (; i < n; i++) {
        __fp16 hv; memcpy(&hv, &in[i], 2);
        out[i] = (float)hv;
    }
}
static inline void fp32_to_fp16_buf(const float* in, uint16_t* out, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t f = vld1q_f32(in + i);
        vst1_f16((__fp16*)(out + i), vcvt_f16_f32(f));
    }
    for (; i < n; i++) {
        __fp16 hv = (__fp16)in[i];
        memcpy(&out[i], &hv, 2);
    }
}
#endif

/* ---- NEON sin (7th-order minimax) ---- */
#ifdef __aarch64__
static inline float32x4_t vsinq_f32(float32x4_t x) {
    const float32x4_t inv_2pi = vdupq_n_f32(0.15915494309f);
    const float32x4_t two_pi  = vdupq_n_f32(6.28318530718f);
    float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_2pi));
    x = vfmsq_f32(x, n, two_pi);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t r = vfmaq_f32(vdupq_n_f32(0.00833333f), vdupq_n_f32(-0.000198413f), x2);
    r = vfmaq_f32(vdupq_n_f32(-0.166666667f), r, x2);
    r = vfmaq_f32(vdupq_n_f32(1.0f), r, x2);
    return vmulq_f32(r, x);
}
#endif

/* ==== CstSin: elementwise sin ==== */
__attribute__((visibility("default")))
int cst_sin_compute(rknn_custom_op_context* ctx,
    rknn_custom_op_tensor* in, uint32_t ni,
    rknn_custom_op_tensor* out, uint32_t no) {
    uint32_t n = tensor_elems(in);
    float* ip = (float*)tensor_data(in);
    float* op = (float*)tensor_data(out);
#ifdef __aarch64__
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(op + i, vsinq_f32(vld1q_f32(ip + i)));
    for (; i < n; i++) op[i] = sinf(ip[i]);
#else
    for (uint32_t i = 0; i < n; i++) op[i] = sinf(ip[i]);
#endif
    return 0;
}

/* ==== CstMul: elementwise multiply (2 inputs) ==== */
__attribute__((visibility("default")))
int cst_mul_compute(rknn_custom_op_context* ctx,
    rknn_custom_op_tensor* in, uint32_t ni,
    rknn_custom_op_tensor* out, uint32_t no) {
    uint32_t n = tensor_elems(&out[0]);
    float* a = (float*)tensor_data(&in[0]);
    float* b = (float*)tensor_data(&in[1]);
    float* o = (float*)tensor_data(&out[0]);

    /* Handle broadcasting: if one input is smaller (constant), broadcast */
    uint32_t na = tensor_elems(&in[0]);
    uint32_t nb = tensor_elems(&in[1]);

    if (na == nb && na == n) {
        /* Same size — elementwise */
#ifdef __aarch64__
        uint32_t i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(o + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        for (; i < n; i++) o[i] = a[i] * b[i];
#else
        for (uint32_t i = 0; i < n; i++) o[i] = a[i] * b[i];
#endif
    } else if (nb < na) {
        /* b is smaller — broadcast b over a */
        for (uint32_t i = 0; i < n; i++) o[i] = a[i] * b[i % nb];
    } else {
        /* a is smaller — broadcast a over b */
        for (uint32_t i = 0; i < n; i++) o[i] = a[i % na] * b[i];
    }
    return 0;
}

/* ==== CstPow: elementwise power (x^exp, exp is constant 2.0) ==== */
__attribute__((visibility("default")))
int cst_pow_compute(rknn_custom_op_context* ctx,
    rknn_custom_op_tensor* in, uint32_t ni,
    rknn_custom_op_tensor* out, uint32_t no) {
    uint32_t n = tensor_elems(&in[0]);
    float* x = (float*)tensor_data(&in[0]);
    float* o = (float*)tensor_data(&out[0]);

    if (ni >= 2) {
        float* exp_p = (float*)tensor_data(&in[1]);
        float exp_val = exp_p[0];
        if (exp_val == 2.0f) {
            /* x² — optimized square */
#ifdef __aarch64__
            uint32_t i = 0;
            for (; i + 4 <= n; i += 4) {
                float32x4_t v = vld1q_f32(x + i);
                vst1q_f32(o + i, vmulq_f32(v, v));
            }
            for (; i < n; i++) o[i] = x[i] * x[i];
#else
            for (uint32_t i = 0; i < n; i++) o[i] = x[i] * x[i];
#endif
        } else {
            for (uint32_t i = 0; i < n; i++) o[i] = powf(x[i], exp_val);
        }
    } else {
        /* Default: x² */
        for (uint32_t i = 0; i < n; i++) o[i] = x[i] * x[i];
    }
    return 0;
}

/* ==== CstAdd: elementwise add (2 inputs, with broadcast) ==== */
__attribute__((visibility("default")))
int cst_add_compute(rknn_custom_op_context* ctx,
    rknn_custom_op_tensor* in, uint32_t ni,
    rknn_custom_op_tensor* out, uint32_t no) {
    uint32_t n = tensor_elems(&out[0]);
    float* a = (float*)tensor_data(&in[0]);
    float* b = (float*)tensor_data(&in[1]);
    float* o = (float*)tensor_data(&out[0]);
    uint32_t na = tensor_elems(&in[0]);
    uint32_t nb = tensor_elems(&in[1]);

    if (na == nb && na == n) {
#ifdef __aarch64__
        uint32_t i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(o + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        for (; i < n; i++) o[i] = a[i] + b[i];
#else
        for (uint32_t i = 0; i < n; i++) o[i] = a[i] + b[i];
#endif
    } else if (nb < na) {
        for (uint32_t i = 0; i < n; i++) o[i] = a[i] + b[i % nb];
    } else {
        for (uint32_t i = 0; i < n; i++) o[i] = a[i % na] + b[i];
    }
    return 0;
}

/* ==== CstInstanceNorm: InstanceNormalization(x, scale, bias, epsilon) ==== */
/* Computes: y = scale * (x - mean) / sqrt(var + eps) + bias
   per channel, across spatial dims. Input shape: [N, C, ...spatial...] */
__attribute__((visibility("default")))
int cst_instance_norm_compute(rknn_custom_op_context* ctx,
    rknn_custom_op_tensor* in, uint32_t ni,
    rknn_custom_op_tensor* out, uint32_t no) {
    float* x     = (float*)tensor_data(&in[0]);
    float* scale  = (ni >= 2) ? (float*)tensor_data(&in[1]) : NULL;
    float* bias   = (ni >= 3) ? (float*)tensor_data(&in[2]) : NULL;
    float* output = (float*)tensor_data(&out[0]);

    uint32_t ndims = in[0].attr.n_dims;
    if (ndims < 2) { memcpy(output, x, in[0].attr.size); return 0; }

    /* Assume NCHW or NC...: dims[0]=N, dims[1]=C, rest=spatial */
    uint32_t N = in[0].attr.dims[0];
    uint32_t C = in[0].attr.dims[1];
    uint32_t spatial = 1;
    for (uint32_t d = 2; d < ndims; d++) spatial *= in[0].attr.dims[d];

    float eps = 1e-5f; /* default epsilon */

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            float* xp = x + (n * C + c) * spatial;
            float* op = output + (n * C + c) * spatial;

            /* Compute mean */
            float sum = 0;
#ifdef __aarch64__
            float32x4_t vsum = vdupq_n_f32(0);
            uint32_t i = 0;
            for (; i + 4 <= spatial; i += 4)
                vsum = vaddq_f32(vsum, vld1q_f32(xp + i));
            sum = vaddvq_f32(vsum);
            for (; i < spatial; i++) sum += xp[i];
#else
            for (uint32_t i = 0; i < spatial; i++) sum += xp[i];
#endif
            float mean = sum / spatial;

            /* Compute variance */
            float var_sum = 0;
#ifdef __aarch64__
            float32x4_t vvar = vdupq_n_f32(0);
            float32x4_t vmean = vdupq_n_f32(mean);
            i = 0;
            for (; i + 4 <= spatial; i += 4) {
                float32x4_t d = vsubq_f32(vld1q_f32(xp + i), vmean);
                vvar = vfmaq_f32(vvar, d, d);
            }
            var_sum = vaddvq_f32(vvar);
            for (; i < spatial; i++) { float d = xp[i] - mean; var_sum += d * d; }
#else
            for (uint32_t i = 0; i < spatial; i++) { float d = xp[i] - mean; var_sum += d * d; }
#endif
            float inv_std = 1.0f / sqrtf(var_sum / spatial + eps);

            float sc = scale ? scale[c] : 1.0f;
            float bi = bias ? bias[c] : 0.0f;
            float a_coeff = sc * inv_std;
            float b_coeff = bi - sc * inv_std * mean;

            /* Apply: y = a * x + b */
#ifdef __aarch64__
            float32x4_t va = vdupq_n_f32(a_coeff);
            float32x4_t vb = vdupq_n_f32(b_coeff);
            i = 0;
            for (; i + 4 <= spatial; i += 4)
                vst1q_f32(op + i, vfmaq_f32(vb, va, vld1q_f32(xp + i)));
            for (; i < spatial; i++) op[i] = a_coeff * xp[i] + b_coeff;
#else
            for (uint32_t i = 0; i < spatial; i++) op[i] = a_coeff * xp[i] + b_coeff;
#endif
        }
    }
    return 0;
}
