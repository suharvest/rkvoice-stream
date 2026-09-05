#ifndef KOKORO_RK3588_BRIDGE_H
#define KOKORO_RK3588_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kokoro_rk3588_handle kokoro_rk3588_handle;

typedef struct {
    uint32_t n_input;
    uint32_t n_output;
    uint32_t input_ndims[6];
    uint32_t input_dims[6][4];
    uint32_t input_fmt[6];
    uint32_t input_floats[6];
    uint32_t output_ndims;
    uint32_t output_dims[4];
    uint32_t output_fmt;
    uint32_t output_floats;
} kokoro_rk3588_contract;

kokoro_rk3588_handle *kokoro_rk3588_create(const char *model_path, int core_mask);
int kokoro_rk3588_query_contract(const kokoro_rk3588_handle *handle,
                                 kokoro_rk3588_contract *contract);
int kokoro_rk3588_run_float32(kokoro_rk3588_handle *handle,
                              const float *const inputs[6],
                              const uint32_t input_floats[6],
                              float *output,
                              uint32_t output_floats);
const char *kokoro_rk3588_last_error(const kokoro_rk3588_handle *handle);
void kokoro_rk3588_destroy(kokoro_rk3588_handle *handle);

#ifdef __cplusplus
}
#endif
#endif
