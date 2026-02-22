# Tiny Neural Network in C++

A minimal neural network engine implemented from scratch in modern C++20.

## Features

- Matrix class (dynamic, row-major)
- Matrix multiplication, transpose, scalar ops
- Activation functions (ReLU, Sigmoid)
- Binary Cross Entropy loss
- Fully connected Dense layer
- Backpropagation with gradient descent
- XOR training example

## Architecture

2 → 8 → 1 neural network

```

Input (2)
↓
Dense (2 → 8) + ReLU
↓
Dense (8 → 1) + Sigmoid

````

## Build

```bash
mkdir build
cd build
cmake ..
make
./tiny_nn
````

## Example Output

```
Epoch 0 Loss: ...
Epoch 9000 Loss: ...
Final predictions:
0.02
0.97
...
```

## Why This Project?

This project demonstrates:

* Neural network math from scratch
* Backpropagation implementation
* Systems-level C++ design
* No external ML libraries

## Future Improvements

* Batch training
* Softmax + multi-class
* MNIST dataset
* SIMD optimization
* OpenMP parallelization
* CUDA implementation

---

Author: Yahia995
License: MIT
