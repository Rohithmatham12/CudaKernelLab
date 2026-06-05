#pragma once

#include <cstddef>
#include <vector>

namespace ckl {

void vector_add(const float* a, const float* b, float* c, std::size_t n);

void reduce_sum(const float* input, float* output, std::size_t n);

void matmul_tiled(const float* a,
                  const float* b,
                  float* c,
                  int m,
                  int n,
                  int k);

void softmax_rows(const float* input, float* output, int rows, int cols);

std::vector<float> cpu_matmul(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int m,
                              int n,
                              int k);

std::vector<float> cpu_softmax_rows(const std::vector<float>& input,
                                    int rows,
                                    int cols);

float cpu_sum(const std::vector<float>& input);

}  // namespace ckl

