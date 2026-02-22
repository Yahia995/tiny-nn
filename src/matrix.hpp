#pragma once

#include <vector>
#include <cstddef>
#include <iostream>

class Matrix {
  public:
    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c);

    double& operator()(size_t r, size_t c);
    double operator()(size_t r, size_t c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator*(const Matrix& other) const; // matrix multiplication

    Matrix transpose() const;

    void print() const;
  
  private:
    std::vector<double> data;
};
