#include <iostream>
#include "matrix.hpp"
#include "activations.hpp"
#include "layer.hpp"

int main() {

  // Input vector (2x1)
  Matrix input(2,1);
  input(0,0) = 1.0;
  input(1,0) = 0.0;

  // Dense layer 2 -> 3 with ReLU
  DenseLayer layer(2, 3, Activation::relu);

  Matrix output = layer.forward(input);

  std::cout << "Layer output:\n";
  output.print();

  return 0;
}
