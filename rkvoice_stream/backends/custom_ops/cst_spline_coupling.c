/**
 * PiperSplineCoupling custom operator for RKNN — NEON accelerated
 *
 * Implements the rational-quadratic (RQ) spline coupling transform from
 * Piper VITS duration predictor normalizing flow (flows.3, flows.5, flows.7).
 *
 * Each flow layer splits channels in half: x_a drives Conv layers that produce
 * spline parameters, x_b is transformed through the RQ spline.  This custom op
 * replaces the 17 problematic ops per layer (NonZero, GatherND, ScatterND,
 * CumSum, etc.) with a single fused CPU kernel.
 *
 * Inputs:
 *   [0] x_b:           (seq_len,) or (1, 1, seq_len)   — values to transform
 *   [1] widths:        (seq_len, K) or (1, 1, seq_len, K) — unnormalized bin widths
 *   [2] heights:       (seq_len, K) or (1, 1, seq_len, K) — unnormalized bin heights
 *   [3] derivatives:   (seq_len, K+1) or (1, 1, seq_len, K+1) — unnormalized derivatives
 *   [4] mask:          (seq_len,) or (1, 1, seq_len)    — bool/float, 1=valid 0=pad
 *
 * Output:
 *   [0] y:             same shape as x_b — transformed values
 *
 * Constants (hardcoded for Piper VITS medium):
 *   K = 10           — number of spline bins
 *   B = 5.0          — tail bound (identity for |x| > B)
 *   min_derivative = 1e-3
 *
 * The transform per element:
 *   1. softmax(widths) * 2B → normalized bin widths (sum = 2B)
 *   2. cumsum(widths) - B → bin edges (cumwidths), range [-B, B]
 *   3. Same for heights → cumheights
 *   4. softplus(derivatives) + min_deriv → positive derivatives
 *   5. searchsorted: find bin k where cumwidths[k] <= x < cumwidths[k+1]
 *   6. RQ spline formula within bin k
 *   7. Identity (passthrough) for |x| > B
 *   8. Zero for masked (padding) positions
 *
 * Build (on ARM64):
 *   gcc -shared -fPIC -O2 -march=armv8-a+simd -o libcstops.so \
 *       cst_spline_coupling.c cst_sin_op.c -lm
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

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

/* ---- Constants ---- */

#define SPLINE_K          10     /* number of bins */
#define SPLINE_K1         11     /* K + 1 */
#define SPLINE_B          5.0f   /* tail bound */
#define SPLINE_2B         10.0f  /* 2 * B */
#define SPLINE_MIN_DERIV  1e-3f  /* minimum derivative */

/* ---- FP16 helpers ---- */

#ifdef __aarch64__
static inline float fp16_to_fp32(uint16_t h) {
    __fp16 hv;
    memcpy(&hv, &h, 2);
    return (float)hv;
}
static inline uint16_t fp32_to_fp16(float f) {
    __fp16 hv = (__fp16)f;
    uint16_t r;
    memcpy(&r, &hv, 2);
    return r;
}
#endif

/* ---- Scalar softmax over K elements ---- */

static inline void softmax_k(const float* in, float* out) {
    float mx = in[0];
    for (int i = 1; i < SPLINE_K; i++) {
        if (in[i] > mx) mx = in[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < SPLINE_K; i++) {
        out[i] = expf(in[i] - mx);
        sum += out[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < SPLINE_K; i++) {
        out[i] *= inv_sum;
    }
}

/* ---- Softplus: log(1 + exp(x)) ---- */

static inline float softplus(float x) {
    if (x > 20.0f) return x;          /* avoid overflow */
    if (x < -20.0f) return expf(x);   /* avoid underflow */
    return logf(1.0f + expf(x));
}

/* ---- RQ spline transform for one element ----
 *
 * Inputs:
 *   x:         scalar value in [-B, B]
 *   widths:    K unnormalized log-widths
 *   heights:   K unnormalized log-heights
 *   derivs:    K+1 unnormalized log-derivatives
 *
 * Returns: transformed value y
 */
static float rq_spline_scalar(float x,
                              const float* widths_raw,
                              const float* heights_raw,
                              const float* derivs_raw) {
    /* 1. Softmax widths and heights, scale to [0, 2B] */
    float w_norm[SPLINE_K], h_norm[SPLINE_K];
    softmax_k(widths_raw, w_norm);
    softmax_k(heights_raw, h_norm);

    /* Scale to 2B */
    for (int i = 0; i < SPLINE_K; i++) {
        w_norm[i] *= SPLINE_2B;
        h_norm[i] *= SPLINE_2B;
    }

    /* 2. CumSum → bin edges, shifted by -B */
    float cumw[SPLINE_K1], cumh[SPLINE_K1];
    cumw[0] = -SPLINE_B;
    cumh[0] = -SPLINE_B;
    for (int i = 0; i < SPLINE_K; i++) {
        cumw[i + 1] = cumw[i] + w_norm[i];
        cumh[i + 1] = cumh[i] + h_norm[i];
    }

    /* 3. Derivatives: softplus + min_deriv */
    float d[SPLINE_K1];
    for (int i = 0; i < SPLINE_K1; i++) {
        d[i] = softplus(derivs_raw[i]) + SPLINE_MIN_DERIV;
    }

    /* 4. Searchsorted: find bin k where cumw[k] <= x < cumw[k+1] */
    int k = 0;
    for (int i = 0; i < SPLINE_K; i++) {
        if (x >= cumw[i + 1]) {
            k = i + 1;
        }
    }
    /* Clamp to valid bin range [0, K-1] */
    if (k >= SPLINE_K) k = SPLINE_K - 1;

    /* 5. RQ formula */
    float w_k = cumw[k + 1] - cumw[k];  /* bin width */
    float h_k = cumh[k + 1] - cumh[k];  /* bin height */
    float s_k = h_k / w_k;              /* slope of linear connection */
    float d_k = d[k];
    float d_k1 = d[k + 1];

    /* Normalized position within bin: xi in [0, 1] */
    float xi = (x - cumw[k]) / w_k;

    /* Numerator: h_k * (s_k * xi^2 + d_k * xi * (1 - xi)) */
    float xi_1mxi = xi * (1.0f - xi);
    float numer = h_k * (s_k * xi * xi + d_k * xi_1mxi);
    /* Denominator: s_k + (d_k1 + d_k - 2*s_k) * xi * (1 - xi) */
    float denom = s_k + (d_k1 + d_k - 2.0f * s_k) * xi_1mxi;

    return cumh[k] + numer / denom;
}

/* ---- Extract flat pointer and element count from tensor ---- */

static inline void* tensor_data(const rknn_custom_op_tensor* t) {
    return (char*)t->mem.virt_addr + t->mem.offset;
}

/* Compute seq_len from tensor shape, ignoring batch/channel dims of size 1.
 * For shapes like (128,), (1,1,128), (1,128) → returns 128.
 * For (128,10), (1,1,128,10) → returns 128 (first non-1 dim or second-to-last).
 */
static uint32_t get_seq_len(const rknn_tensor_attr* attr) {
    uint32_t nd = attr->n_dims;
    if (nd == 0) return 0;

    /* For multi-dim tensors (widths/heights/derivs), seq_len is second-to-last dim */
    if (nd >= 2) {
        /* Find the meaningful dimensions by skipping leading 1s */
        uint32_t first_real = 0;
        while (first_real < nd - 1 && attr->dims[first_real] == 1)
            first_real++;
        return attr->dims[first_real];
    }
    return attr->dims[0];
}

/* ---- Main compute function ---- */

__attribute__((visibility("default")))
int cst_spline_coupling_compute(rknn_custom_op_context* op_ctx,
                                rknn_custom_op_tensor* inputs,  uint32_t n_inputs,
                                rknn_custom_op_tensor* outputs, uint32_t n_outputs)
{
    if (n_inputs < 5 || n_outputs < 1)
        return -1;

    const rknn_tensor_attr* x_attr = &inputs[0].attr;
    uint32_t seq_len = get_seq_len(x_attr);
    if (seq_len == 0) return -1;

    int is_fp16 = (x_attr->type == RKNN_TENSOR_FLOAT16);

    /* Temporary FP32 buffers on stack (seq_len <= 256 typical) */
    float x_buf[256];
    float w_buf[256 * SPLINE_K];
    float h_buf[256 * SPLINE_K];
    float d_buf[256 * SPLINE_K1];
    float mask_buf[256];
    float y_buf[256];

    if (seq_len > 256) return -2;  /* safety: stack overflow guard */

    /* ---- Load x_b ---- */
    void* x_ptr = tensor_data(&inputs[0]);
    if (is_fp16) {
#ifdef __aarch64__
        const uint16_t* hp = (const uint16_t*)x_ptr;
        for (uint32_t i = 0; i < seq_len; i++)
            x_buf[i] = fp16_to_fp32(hp[i]);
#else
        return -1;
#endif
    } else {
        memcpy(x_buf, x_ptr, seq_len * sizeof(float));
    }

    /* ---- Load widths (seq_len, K) ---- */
    void* w_ptr = tensor_data(&inputs[1]);
    if (is_fp16) {
#ifdef __aarch64__
        const uint16_t* hp = (const uint16_t*)w_ptr;
        uint32_t n = seq_len * SPLINE_K;
        for (uint32_t i = 0; i < n; i++)
            w_buf[i] = fp16_to_fp32(hp[i]);
#else
        return -1;
#endif
    } else {
        memcpy(w_buf, w_ptr, seq_len * SPLINE_K * sizeof(float));
    }

    /* ---- Load heights (seq_len, K) ---- */
    void* h_ptr = tensor_data(&inputs[2]);
    if (is_fp16) {
#ifdef __aarch64__
        const uint16_t* hp = (const uint16_t*)h_ptr;
        uint32_t n = seq_len * SPLINE_K;
        for (uint32_t i = 0; i < n; i++)
            h_buf[i] = fp16_to_fp32(hp[i]);
#else
        return -1;
#endif
    } else {
        memcpy(h_buf, h_ptr, seq_len * SPLINE_K * sizeof(float));
    }

    /* ---- Load derivatives (seq_len, K+1) ---- */
    void* d_ptr = tensor_data(&inputs[3]);
    if (is_fp16) {
#ifdef __aarch64__
        const uint16_t* hp = (const uint16_t*)d_ptr;
        uint32_t n = seq_len * SPLINE_K1;
        for (uint32_t i = 0; i < n; i++)
            d_buf[i] = fp16_to_fp32(hp[i]);
#else
        return -1;
#endif
    } else {
        memcpy(d_buf, d_ptr, seq_len * SPLINE_K1 * sizeof(float));
    }

    /* ---- Load mask ---- */
    void* m_ptr = tensor_data(&inputs[4]);
    if (inputs[4].attr.type == RKNN_TENSOR_FLOAT16) {
#ifdef __aarch64__
        const uint16_t* hp = (const uint16_t*)m_ptr;
        for (uint32_t i = 0; i < seq_len; i++)
            mask_buf[i] = fp16_to_fp32(hp[i]);
#else
        return -1;
#endif
    } else {
        /* Could be float32 or int/bool — treat as float */
        const float* fp = (const float*)m_ptr;
        for (uint32_t i = 0; i < seq_len; i++)
            mask_buf[i] = fp[i];
    }

    /* ---- Transform each position ---- */
    for (uint32_t i = 0; i < seq_len; i++) {
        if (mask_buf[i] < 0.5f) {
            /* Padding position: output zero */
            y_buf[i] = 0.0f;
            continue;
        }

        float x = x_buf[i];

        /* Tail: identity for |x| > B */
        if (x <= -SPLINE_B || x >= SPLINE_B) {
            y_buf[i] = x;
            continue;
        }

        /* In-range: apply RQ spline */
        y_buf[i] = rq_spline_scalar(x,
                                    &w_buf[i * SPLINE_K],
                                    &h_buf[i * SPLINE_K],
                                    &d_buf[i * SPLINE_K1]);
    }

    /* ---- Write output ---- */
    void* out_ptr = tensor_data(&outputs[0]);
    if (is_fp16) {
#ifdef __aarch64__
        uint16_t* hp = (uint16_t*)out_ptr;
        for (uint32_t i = 0; i < seq_len; i++)
            hp[i] = fp32_to_fp16(y_buf[i]);
#else
        return -1;
#endif
    } else {
        memcpy(out_ptr, y_buf, seq_len * sizeof(float));
    }

    return 0;
}
