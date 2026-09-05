#include "kokoro_rk3588_bridge.h"

#include <rknn_api.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

struct kokoro_rk3588_handle {
    rknn_context ctx = 0;
    kokoro_rk3588_contract contract{};
    std::string error;
    mutable std::mutex mutex;
};

static int fail(kokoro_rk3588_handle *h, const char *message) {
    if (h) h->error = message;
    return -1;
}

static bool finite_array(const float *p, uint32_t n) {
    if (!p) return false;
    for (uint32_t i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}

extern "C" kokoro_rk3588_handle *kokoro_rk3588_create(const char *path, int core_mask) {
    if (!path) return nullptr;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return nullptr;
    const std::streamsize size = f.tellg();
    if (size <= 0) return nullptr;
    f.seekg(0);
    std::vector<uint8_t> model(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char *>(model.data()), size)) return nullptr;
    auto *h = new kokoro_rk3588_handle();
    int ret = rknn_init(&h->ctx, model.data(), model.size(), 0, nullptr);
    if (ret != 0) { delete h; return nullptr; }
    if (core_mask != 0) ret = rknn_set_core_mask(h->ctx, static_cast<rknn_core_mask>(core_mask));
    if (ret != 0) { rknn_destroy(h->ctx); delete h; return nullptr; }
    rknn_input_output_num io{};
    if ((ret = rknn_query(h->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io))) != 0 || io.n_input == 0 || io.n_input > 6 || io.n_output != 1) {
        rknn_destroy(h->ctx); delete h; return nullptr;
    }
    h->contract.n_input = io.n_input;
    h->contract.n_output = io.n_output;
    for (uint32_t i = 0; i < h->contract.n_input; ++i) {
        rknn_tensor_attr a{}; a.index = i;
        if (rknn_query(h->ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)) != 0 || a.n_dims > 4 || a.n_dims == 0) {
            rknn_destroy(h->ctx); delete h; return nullptr;
        }
        h->contract.input_ndims[i] = a.n_dims;
        h->contract.input_fmt[i] = static_cast<uint32_t>(a.fmt);
        uint64_t count = 1;
        for (uint32_t d = 0; d < a.n_dims; ++d) { h->contract.input_dims[i][d] = a.dims[d]; count *= a.dims[d]; }
        if (count == 0 || count > std::numeric_limits<uint32_t>::max()) { rknn_destroy(h->ctx); delete h; return nullptr; }
        h->contract.input_floats[i] = static_cast<uint32_t>(count);
    }
    rknn_tensor_attr a{}; a.index = 0;
    if (rknn_query(h->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) != 0 || a.n_dims > 4 || a.n_dims == 0) {
        rknn_destroy(h->ctx); delete h; return nullptr;
    }
    h->contract.output_ndims = a.n_dims;
    h->contract.output_fmt = static_cast<uint32_t>(a.fmt);
    uint64_t count = 1;
    for (uint32_t d = 0; d < a.n_dims; ++d) { h->contract.output_dims[d] = a.dims[d]; count *= a.dims[d]; }
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) { rknn_destroy(h->ctx); delete h; return nullptr; }
    h->contract.output_floats = static_cast<uint32_t>(count);
    return h;
}

extern "C" int kokoro_rk3588_query_contract(const kokoro_rk3588_handle *h, kokoro_rk3588_contract *out) {
    if (!h || !out) return -1;
    std::lock_guard<std::mutex> lock(h->mutex);
    *out = h->contract;
    return 0;
}

extern "C" int kokoro_rk3588_run_float32(kokoro_rk3588_handle *h, const float *const inputs[6], const uint32_t sizes[6], float *output, uint32_t output_floats) {
    if (!h || !inputs || !sizes || !output || output_floats < h->contract.output_floats) return fail(h, "invalid run arguments");
    std::lock_guard<std::mutex> lock(h->mutex);
    rknn_input in[6]{};
    for (uint32_t i = 0; i < h->contract.n_input; ++i) {
        if (sizes[i] != h->contract.input_floats[i]) return fail(h, "input element count mismatch");
        if (!finite_array(inputs[i], sizes[i])) return fail(h, "input null/nonfinite");
        in[i].index = i; in[i].buf = const_cast<float *>(inputs[i]); in[i].size = sizes[i] * sizeof(float);
        // Host buffers are float32 while the compiled graphs may have a
        // different native type.  pass_through=0 is required so librknnrt
        // performs the documented conversion and owns its staging buffer.
        in[i].pass_through = 0; in[i].type = RKNN_TENSOR_FLOAT32;
        in[i].fmt = static_cast<rknn_tensor_format>(h->contract.input_fmt[i]);
    }
    int ret = rknn_inputs_set(h->ctx, h->contract.n_input, in);
    if (ret != 0) return fail(h, "rknn_inputs_set failed");
    ret = rknn_run(h->ctx, nullptr);
    if (ret != 0) return fail(h, "rknn_run failed");
    rknn_output out{}; out.index = 0; out.want_float = 1; out.is_prealloc = 0;
    ret = rknn_outputs_get(h->ctx, 1, &out, nullptr);
    if (ret != 0) return fail(h, "rknn_outputs_get failed");
    const uint32_t bytes = h->contract.output_floats * sizeof(float);
    if (!out.buf || out.size < bytes) { rknn_outputs_release(h->ctx, 1, &out); return fail(h, "short output"); }
    std::memcpy(output, out.buf, bytes);
    ret = rknn_outputs_release(h->ctx, 1, &out);
    if (ret != 0) return fail(h, "rknn_outputs_release failed");
    if (!finite_array(output, h->contract.output_floats)) return fail(h, "output nonfinite");
    h->error.clear();
    return 0;
}

extern "C" const char *kokoro_rk3588_last_error(const kokoro_rk3588_handle *h) {
    return h ? h->error.c_str() : "null handle";
}

extern "C" void kokoro_rk3588_destroy(kokoro_rk3588_handle *h) {
    if (!h) return;
    // Caller protocol: destroy must not run concurrently with run/query.
    // Release the mutex scope before deleting the handle.
    {
        std::lock_guard<std::mutex> lock(h->mutex);
        if (h->ctx) rknn_destroy(h->ctx);
        h->ctx = 0;
    }
    delete h;
}
