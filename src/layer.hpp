#pragma once

#include "matrix.hpp"
#include <functional>

class DenseLayer {
  private:
    Matrix weights;
    Matrix bias;

    std::function<double(double)> activation;

  public:
    DenseLayer(size_t input_size,
               size_t output_size,
               std::function<double(double)> act);

    Matrix forward(const Matrix& input);
};
