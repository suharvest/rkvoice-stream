#ifndef KOKORO_CONVONLY_NATIVE_H
#define KOKORO_CONVONLY_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kokoro_convonly_handle kokoro_convonly_handle;

/* RK3576 all-FP16 native tail.  Every pointer is borrowed for the call only. */
kokoro_convonly_handle *kokoro_convonly_create(
    const char *model_root, const char *merge_path,
    const float *slopes, uint32_t slope_count, uint32_t max_n,
    char *error, uint32_t error_bytes);

int kokoro_convonly_run(
    kokoro_convonly_handle *handle, uint32_t n,
    const float *j1, uint32_t j1_count,
    const float *gamma, uint32_t gamma_count,
    const float *beta, uint32_t beta_count,
    float *output, uint32_t output_count,
    char *error, uint32_t error_bytes);

/* The caller must quiesce run calls before destroy. */
void kokoro_convonly_destroy(kokoro_convonly_handle *handle);

#ifdef __cplusplus
}
#endif
#endif
