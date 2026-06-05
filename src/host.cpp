#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ckl {

std::vector<float> cpu_matmul(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int m,
                              int n,
                              int k) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        acc += a[row * k + inner] * b[inner * n + col];
      }
      c[row * n + col] = acc;
    }
  }
  return c;
}

std::vector<float> cpu_softmax_rows(const std::vector<float>& input,
                                    int rows,
                                    int cols) {
  std::vector<float> output(input.size());
  for (int row = 0; row < rows; ++row) {
    const auto begin = input.begin() + static_cast<std::ptrdiff_t>(row * cols);
    const auto end = begin + cols;
    const float max_value = *std::max_element(begin, end);
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float value = std::exp(input[row * cols + col] - max_value);
      output[row * cols + col] = value;
      sum += value;
    }
    for (int col = 0; col < cols; ++col) {
      output[row * cols + col] /= sum;
    }
  }
  return output;
}

float cpu_sum(const std::vector<float>& input) {
  return std::accumulate(input.begin(), input.end(), 0.0f);
}

}  // namespace ckl

