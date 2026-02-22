#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "activations.hpp"
#include "layer.hpp"
#include "loss.hpp"

int main() {

  // XOR dataset
  std::vector<std::pair<Matrix, Matrix>> dataset;

  Matrix x1(2,1); x1(0,0)=0; x1(1,0)=0;
  Matrix y1(1,1); y1(0,0)=0;
  dataset.push_back({x1,y1});

  Matrix x2(2,1); x2(0,0)=0; x2(1,0)=1;
  Matrix y2(1,1); y2(0,0)=1;
  dataset.push_back({x2,y2});

  Matrix x3(2,1); x3(0,0)=1; x3(1,0)=0;
  Matrix y3(1,1); y3(0,0)=1;
  dataset.push_back({x3,y3});

  Matrix x4(2,1); x4(0,0)=1; x4(1,0)=1;
  Matrix y4(1,1); y4(0,0)=0;
  dataset.push_back({x4,y4});

  // Network
  DenseLayer layer1(
    2, 8,
    Activation::relu,
    Activation::relu_derivative
  );

  DenseLayer layer2(
    8, 1,
    Activation::sigmoid,
    Activation::sigmoid_derivative
  );

  double learning_rate = 0.1;
  int epochs = 10000;

  for (int epoch = 0; epoch < epochs; ++epoch) {

    double total_loss = 0.0;

    for (auto& sample : dataset) {

      Matrix input = sample.first;
      Matrix target = sample.second;

      // Forward
      Matrix h = layer1.forward(input);
      Matrix output = layer2.forward(h);

      // Loss
      total_loss += Loss::binary_cross_entropy(output, target);

      // Backward
      Matrix grad_loss =
        Loss::binary_cross_entropy_derivative(output, target);

      Matrix grad_l2 = layer2.backward(grad_loss, learning_rate);
      layer1.backward(grad_l2, learning_rate);
    }

    if (epoch % 1000 == 0)
      std::cout << "Epoch " << epoch
                << " Loss: " << total_loss / dataset.size()
                << std::endl;
  }

  // Test trained model
  std::cout << "\nFinal predictions:\n";

  for (auto& sample : dataset) {
    Matrix h = layer1.forward(sample.first);
    Matrix out = layer2.forward(h);
    out.print();
    std::cout << "------\n";
  }

  return 0;
}
