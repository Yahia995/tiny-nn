#pragma once

#include <cmath>
#include "matrix.hpp"

namespace Loss {

  // Binary Cross Entropy (single output case)
  inline double binary_cross_entropy(const Matrix& y_pred,
                                     const Matrix& y_true) {
    double epsilon = 1e-9; // avoid log(0)
    double loss = 0.0;

    for (size_t i = 0; i < y_pred.rows; ++i) {
      double p = y_pred(i, 0);
      double y = y_true(i, 0);

      loss += - ( y * std::log(p + epsilon) 
              + (1 - y) * std::log(1 - p + epsilon) );
    }

    return loss / y_pred.rows;
  }

  // Derivative of BCE w.r.t prediction
  inline Matrix binary_cross_entropy_derivative(const Matrix& y_pred,
                                                const Matrix& y_true) {
    double epsilon = 1e-9;

    Matrix grad(y_pred.rows, 1);

    for (size_t i = 0; i < y_pred.rows; ++i) {
      double p = y_pred(i, 0);
      double y = y_true(i, 0);

      grad(i, 0) =
        (p - y) / ((p + epsilon) * (1 - p + epsilon));
    }

    return grad;
  }

}
