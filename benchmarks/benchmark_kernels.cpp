#include "kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iomanip>
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

std::vector<float> random_vector(std::size_t n) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(n);
  for (auto& value : out) {
    value = dist(rng);
  }
  return out;
}

float time_ms(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  check_cuda(cudaDeviceSynchronize(), "warmup sync");
  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  check_cuda(cudaDeviceSynchronize(), "benchmark sync");
  const auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<float, std::milli>(end - start).count() / iters;
}

void benchmark_vector_add() {
  const std::size_t n = 1 << 24;
  const auto host = random_vector(n);
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&a), n * sizeof(float)),
             "malloc a");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&b), n * sizeof(float)),
             "malloc b");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&c), n * sizeof(float)),
             "malloc c");
  check_cuda(cudaMemcpy(a, host.data(), n * sizeof(float), cudaMemcpyHostToDevice),
             "copy a");
  check_cuda(cudaMemcpy(b, host.data(), n * sizeof(float), cudaMemcpyHostToDevice),
             "copy b");

  const float ms = time_ms([&] { ckl::vector_add(a, b, c, n); }, 5, 50);
  const double gb = static_cast<double>(n) * sizeof(float) * 3.0 / 1e9;
  std::cout << "vector_add: " << ms << " ms, " << gb / (ms / 1e3)
            << " GB/s\n";

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

void benchmark_matmul() {
  const int m = 1024;
  const int n = 1024;
  const int k = 1024;
  const auto ha = random_vector(static_cast<std::size_t>(m) * k);
  const auto hb = random_vector(static_cast<std::size_t>(k) * n);
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&a),
                        ha.size() * sizeof(float)),
             "malloc a");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&b),
                        hb.size() * sizeof(float)),
             "malloc b");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&c),
                        static_cast<std::size_t>(m) * n * sizeof(float)),
             "malloc c");
  check_cuda(cudaMemcpy(a, ha.data(), ha.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy a");
  check_cuda(cudaMemcpy(b, hb.data(), hb.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy b");

  const float ms = time_ms([&] { ckl::matmul_tiled(a, b, c, m, n, k); }, 3, 20);
  const double tflops = 2.0 * m * n * k / (ms / 1e3) / 1e12;
  std::cout << "matmul_tiled 1024x1024: " << ms << " ms, " << tflops
            << " TFLOP/s\n";

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

}  // namespace

int main() {
  try {
    std::cout << std::fixed << std::setprecision(3);
    benchmark_vector_add();
    benchmark_matmul();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
