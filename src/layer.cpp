#include "layer.hpp"
#include "activations.hpp"

#include <random>

DenseLayer::DenseLayer(size_t input_size,
                       size_t output_size,
                       std::function<double(double)> act,
                       std::function<double(double)> act_deriv)
  : weights(output_size, input_size),
    bias(output_size, 1),
    activation(act),
    activation_derivative(act_deriv)
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
  input_cache = input;

  z_cache = weights * input;
  z_cache = z_cache + bias;

  return Activation::apply(z_cache, activation);
}

Matrix DenseLayer::backward(const Matrix& grad_output,
                            double learning_rate) {
  // Apply activation derivative to z
  Matrix activation_grad =
    Activation::apply(z_cache, activation_derivative);

  // Element-wise multiply
  Matrix grad_z(z_cache.rows, z_cache.cols);

  for (size_t i = 0; i < grad_z.rows; ++i) {
    for (size_t j = 0; j < grad_z.cols; ++j)
      grad_z(i,j) = grad_output(i,j) * activation_grad(i,j);
  }

  // Gradients
  Matrix grad_w = grad_z * input_cache.transpose();
  Matrix grad_b = grad_z;

  Matrix grad_input = weights.transpose() * grad_z;

  // Update weights
  weights = weights - grad_w * learning_rate;
  bias    = bias - grad_b * learning_rate;

  return grad_input;
}
