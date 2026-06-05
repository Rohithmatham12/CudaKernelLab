#include "kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

bool near(float a, float b, float tol = 1e-3f) {
  return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b));
}

std::vector<float> random_vector(std::size_t n) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(n);
  for (auto& value : out) {
    value = dist(rng);
  }
  return out;
}

void copy_to_device(float** ptr, const std::vector<float>& host) {
  check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr),
                        host.size() * sizeof(float)),
             "cudaMalloc");
  check_cuda(cudaMemcpy(*ptr,
                        host.data(),
                        host.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy host to device");
}

std::vector<float> copy_to_host(float* ptr, std::size_t n) {
  std::vector<float> host(n);
  check_cuda(cudaMemcpy(host.data(),
                        ptr,
                        n * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copy device to host");
  return host;
}

void test_vector_add() {
  const std::size_t n = 1 << 20;
  const auto a = random_vector(n);
  const auto b = random_vector(n);

  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  copy_to_device(&da, a);
  copy_to_device(&db, b);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&dc), n * sizeof(float)),
             "cudaMalloc output");

  ckl::vector_add(da, db, dc, n);
  check_cuda(cudaDeviceSynchronize(), "sync vector_add");
  const auto c = copy_to_host(dc, n);

  for (std::size_t i = 0; i < n; ++i) {
    if (!near(c[i], a[i] + b[i])) {
      throw std::runtime_error("vector_add mismatch");
    }
  }

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}

void test_reduce_sum() {
  const std::size_t n = 1 << 18;
  const auto input = random_vector(n);
  float* din = nullptr;
  float* dout = nullptr;
  copy_to_device(&din, input);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&dout), sizeof(float)),
             "cudaMalloc reduction output");

  ckl::reduce_sum(din, dout, n);
  check_cuda(cudaDeviceSynchronize(), "sync reduce_sum");
  const auto out = copy_to_host(dout, 1);
  const float expected = ckl::cpu_sum(input);

  if (!near(out[0], expected, 1e-2f)) {
    throw std::runtime_error("reduce_sum mismatch");
  }

  cudaFree(din);
  cudaFree(dout);
}

void test_matmul_tiled() {
  const int m = 65;
  const int n = 73;
  const int k = 31;
  const auto a = random_vector(static_cast<std::size_t>(m) * k);
  const auto b = random_vector(static_cast<std::size_t>(k) * n);
  const auto expected = ckl::cpu_matmul(a, b, m, n, k);

  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  copy_to_device(&da, a);
  copy_to_device(&db, b);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&dc),
                        expected.size() * sizeof(float)),
             "cudaMalloc matmul output");

  ckl::matmul_tiled(da, db, dc, m, n, k);
  check_cuda(cudaDeviceSynchronize(), "sync matmul_tiled");
  const auto actual = copy_to_host(dc, expected.size());

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!near(actual[i], expected[i], 1e-2f)) {
      throw std::runtime_error("matmul_tiled mismatch");
    }
  }

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
}

void test_softmax_rows() {
  const int rows = 17;
  const int cols = 257;
  const auto input = random_vector(static_cast<std::size_t>(rows) * cols);
  const auto expected = ckl::cpu_softmax_rows(input, rows, cols);

  float* din = nullptr;
  float* dout = nullptr;
  copy_to_device(&din, input);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&dout),
                        input.size() * sizeof(float)),
             "cudaMalloc softmax output");

  ckl::softmax_rows(din, dout, rows, cols);
  check_cuda(cudaDeviceSynchronize(), "sync softmax_rows");
  const auto actual = copy_to_host(dout, input.size());

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!near(actual[i], expected[i], 1e-3f)) {
      throw std::runtime_error("softmax_rows mismatch");
    }
  }

  cudaFree(din);
  cudaFree(dout);
}

}  // namespace

int main() {
  try {
    int devices = 0;
    check_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
      std::cerr << "No CUDA devices found\n";
      return 77;
    }

    test_vector_add();
    test_reduce_sum();
    test_matmul_tiled();
    test_softmax_rows();
    std::cout << "All CUDA kernel tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
