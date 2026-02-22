#include <iostream>
#include "matrix.hpp"
#include "activations.hpp"
#include "layer.hpp"

int main() {

  Matrix input(2,1);
  input(0,0) = 1.0;
  input(1,0) = 0.0;

  DenseLayer layer(
    2, 1,
    Activation::sigmoid,
    Activation::sigmoid_derivative
  );

  Matrix output = layer.forward(input);

  Matrix grad(1,1);
  grad(0,0) = 1.0;

  layer.backward(grad, 0.01);

  std::cout << "Backward pass executed." << std::endl;

  return 0;
}
