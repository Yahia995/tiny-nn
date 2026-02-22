#pragma once

#include "matrix.hpp"
#include <functional>

class DenseLayer {
  private:
    Matrix weights;
    Matrix bias;

    Matrix input_cache;
    Matrix z_cache;

    std::function<double(double)> activation;
    std::function<double(double)> activation_derivative;

  public:
    DenseLayer(size_t input_size,
               size_t output_size,
               std::function<double(double)> act,
               std::function<double(double)> act_deriv);

    Matrix forward(const Matrix& input);

    Matrix backward(const Matrix& grad_output, double learning_rate);
};
