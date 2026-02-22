#include "layer.hpp"
#include "activations.hpp"

#include <random>

DenseLayer::DenseLayer(size_t input_size,
                       size_t output_size,
                       std::function<double(double)> act)
  : weights(output_size, input_size),
    bias(output_size, 1),
    activation(act)
{
  // Xavier-style small random initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(input_size));

  for (size_t i = 0; i < weights.rows; ++i) {
    for (size_t j = 0; j < weights.cols; ++j)
      weights(i, j) = dist(gen);
  }

  for (size_t i = 0; i < bias.rows; ++i)
    bias(i, 0) = 0.0;
}

Matrix DenseLayer::forward(const Matrix& input) {
  Matrix z = weights * input;
  z = z + bias;

  Matrix activated = Activation::apply(z, activation);
  return activated;
}
