# CudaKernelLab

CUDA kernel programming exercises with correctness tests and simple benchmark
drivers. The project focuses on core GPU programming patterns:

- tiled matrix multiplication with shared memory
- block and warp reductions
- row-wise softmax with numerically stable max/subtract/exp/sum
- memory bandwidth benchmarking for vector operations

## Requirements

- NVIDIA GPU
- CUDA Toolkit 12.x or newer
- CMake 3.24+
- C++17 compiler supported by CUDA

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you need to target a specific architecture:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## Run Benchmarks

```bash
./build/benchmark_kernels
```

The benchmark prints elapsed time, effective bandwidth, and GEMM throughput for
the included kernels.

