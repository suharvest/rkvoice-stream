# RK3576 native Conv-only tail

This package exposes the bounded C ABI required by the `kokoro_convonly` backend:
`create(root, merge, slopes, count, max_N, error_buffer)`, `run(handle, N,
inputs+exact counts, output+exact count, error_buffer)`, and `destroy(handle)`.
The handle owns persistent scratch and serializes calls. Scratch is resized only
after acquiring the run mutex, so length changes occur while idle. Destruction
must be called only after the caller has stopped submitting work and joined all
callers. The Python wrapper holds its lifecycle lock across run and destroy;
the C destroy function requires external quiescence and cannot validate a stale pointer.

The production build uses the real RKNN bridge, 18 Conv contexts and one merge
context. Three persistent workers retain masks `1,2,1`; every submission is
drained even after another branch fails. Normalization, FiLM, PReLU, halo
stitching and per-unit residuals retain the approved source arithmetic.
Missing SDK headers/runtime cause a build failure, not a synthetic fallback.

Host tests compile the same production source against `tests/stub_bridge.cpp`
in a pytest temporary directory. Only Conv/merge calls are substituted; the
real CPU arithmetic is compared with NumPy at lengths 17/8193/12001/38401.
This does not establish RKNN device correctness or performance. The host stub
is test-only and is never packaged or loaded by the runtime.

Build locally with:

```sh
make -C native/kokoro_convonly
```

The RK3576 qualification build uses the same source with the existing target
toolchain and RKNN runtime, for example:

```sh
make -C native/kokoro_convonly \
  CXX=aarch64-linux-gnu-g++ \
  CXXFLAGS='-O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -march=armv8-a+simd -ffp-contract=off -fPIC' \
  RKNN_INC=/path/to/rknn/include RKNN_LIB=/path/to/rknn/lib
```

The copied bridge uses `rknn_inputs_set`, `rknn_run`,
`rknn_outputs_get`, copy, and `rknn_outputs_release` for every branch and merge
context. Any branch submission error is followed by joining/draining all
submitted branches before returning the error.

Build qualification on RK3576 must record the existing aarch64 RKNN SDK header,
`librknnrt`, compiler, flags, and source/header/runtime hashes. No SDK is
downloaded by this package.

Provenance: compute source derives from frozen `kokoro_native_full.cpp` SHA256
`5a92124f1266e8af93c0d1a4afbb00012b3407089dbebe1ed20320e9e7b300de`.
Bridge source/header are byte copies of the approved probe, SHA256 respectively
`45951542d162b94d91209af17f2152bf5bc2d63b570611b9bf7d2648a5cfecca` and
`2bdde2b971c3e7e6ed5af614e20ba36f5673e96103ab1eb2d24d72a1c57e0d47`.
New API/lifecycle code needs its own device qualification; historical binary
results do not qualify the new shared library.
