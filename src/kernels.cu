#include "kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kThreads = 256;

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void reduce_sum_kernel(const float* input,
                                  float* partials,
                                  std::size_t n) {
  __shared__ float shared[kThreads];
  const unsigned tid = threadIdx.x;
  const std::size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float sum = 0.0f;
  if (i < n) {
    sum += input[i];
  }
  if (i + blockDim.x < n) {
    sum += input[i + blockDim.x];
  }
  shared[tid] = sum;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile float* vshared = shared;
    vshared[tid] += vshared[tid + 32];
    vshared[tid] += vshared[tid + 16];
    vshared[tid] += vshared[tid + 8];
    vshared[tid] += vshared[tid + 4];
    vshared[tid] += vshared[tid + 2];
    vshared[tid] += vshared[tid + 1];
  }

  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void matmul_tiled_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
  __shared__ float as[kTile][kTile];
  __shared__ float bs[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float acc = 0.0f;

  for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
    const int tiled_col = tile * kTile + threadIdx.x;
    const int tiled_row = tile * kTile + threadIdx.y;

    as[threadIdx.y][threadIdx.x] =
        (row < m && tiled_col < k) ? a[row * k + tiled_col] : 0.0f;
    bs[threadIdx.y][threadIdx.x] =
        (tiled_row < k && col < n) ? b[tiled_row * n + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < kTile; ++i) {
      acc += as[threadIdx.y][i] * bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

__global__ void softmax_rows_kernel(const float* input,
                                    float* output,
                                    int rows,
                                    int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  float local_max = -INFINITY;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, input[row * cols + col]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = shared[0];

  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_sum += expf(input[row * cols + col] - row_max);
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float row_sum = shared[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    output[row * cols + col] = expf(input[row * cols + col] - row_max) / row_sum;
  }
}

}  // namespace

namespace ckl {

void vector_add(const float* a, const float* b, float* c, std::size_t n) {
  const dim3 block(kThreads);
  const dim3 grid((n + block.x - 1) / block.x);
  vector_add_kernel<<<grid, block>>>(a, b, c, n);
  check_cuda(cudaGetLastError(), "vector_add launch");
}

void reduce_sum(const float* input, float* output, std::size_t n) {
  if (n == 0) {
    check_cuda(cudaMemset(output, 0, sizeof(float)), "zero reduction output");
    return;
  }

  std::size_t blocks = (n + kThreads * 2 - 1) / (kThreads * 2);
  float* partials_a = nullptr;
  float* partials_b = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&partials_a),
                        blocks * sizeof(float)),
             "allocate partials_a");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&partials_b),
                        blocks * sizeof(float)),
             "allocate partials_b");

  const float* current_input = input;
  std::size_t current_n = n;
  float* current_output = partials_a;
  bool use_a = true;

  while (true) {
    blocks = (current_n + kThreads * 2 - 1) / (kThreads * 2);
    reduce_sum_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(
        current_input, current_output, current_n);
    check_cuda(cudaGetLastError(), "reduce_sum launch");

    if (blocks == 1) {
      check_cuda(cudaMemcpy(output,
                            current_output,
                            sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "copy reduction output");
      break;
    }
    current_input = current_output;
    current_n = blocks;
    use_a = !use_a;
    current_output = use_a ? partials_a : partials_b;
  }

  check_cuda(cudaFree(partials_a), "free partials_a");
  check_cuda(cudaFree(partials_b), "free partials_b");
}

void matmul_tiled(const float* a,
                  const float* b,
                  float* c,
                  int m,
                  int n,
                  int k) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  matmul_tiled_kernel<<<grid, block>>>(a, b, c, m, n, k);
  check_cuda(cudaGetLastError(), "matmul_tiled launch");
}

void softmax_rows(const float* input, float* output, int rows, int cols) {
  const int threads = 256;
  softmax_rows_kernel<<<rows, threads, threads * sizeof(float)>>>(
      input, output, rows, cols);
  check_cuda(cudaGetLastError(), "softmax_rows launch");
}

}  // namespace ckl
