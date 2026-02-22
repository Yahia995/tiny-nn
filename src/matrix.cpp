#include <stdexcept>
#include "matrix.hpp"

Matrix::Matrix(size_t r, size_t c)
  : rows(r), cols(c), data(r * c, 0.0) {}

double& Matrix::operator()(size_t r, size_t c) {
  return data[r * cols + c];
}

double Matrix::operator()(size_t r, size_t c) const {
  return data[r * cols + c];
}

Matrix Matrix::operator+(const Matrix& other) const {
  if (rows != other.rows || cols != other.cols)
    throw std::invalid_argument("Matrix dimensions must match for addition");

  Matrix result(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i)
    result.data[i] = data[i] + other.data[i];

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  if (rows != other.rows || cols != other.cols)
    throw std::invalid_argument("Matrix dimensions must match for subtraction");

  Matrix result(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i)
    result.data[i] = data[i] - other.data[i];

  return result;
}

Matrix Matrix::operator*(double scalar) const {
  Matrix result(rows, cols);

  for (size_t i = 0; i < rows * cols; ++i)
    result.data[i] = data[i] * scalar;

  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  if (cols != other.rows)
    throw std::invalid_argument("Invalid dimensions for matrix multiplication");

  Matrix result(rows, other.cols);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < other.cols; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < cols; ++k)
        sum += (*this)(i, k) * other(k, j);
      result(i, j) = sum;
    }
  }

  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(cols, rows);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j)
      result(j, i) = (*this)(i, j);
  }

  return result;
}

void Matrix::print() const {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j)
      std::cout << (*this)(i, j) << " ";
    std::cout << "\n";
  }
}
