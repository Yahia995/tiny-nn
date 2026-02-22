#pragma once

#include <cmath>
#include "matrix.hpp"

namespace Activation {

  // --- Scalar versions ---

  inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
  }

  inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
  }

  inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
  }

  inline double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
  }

  // --- Matrix versions (element-wise) ---

  template<typename Func>
  inline Matrix apply(const Matrix& m, Func func) {
    Matrix result(m.rows, m.cols);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j)
            result(i, j) = func(m(i, j));
    }

    return result;
  }
}
