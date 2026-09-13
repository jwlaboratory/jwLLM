#pragma once

#include <vector>

class Matrix
{
public:
    int rows;
    int cols;
    std::vector<float> data;

    Matrix();
    Matrix(int rows, int cols, std::vector<float> passed_data);
    Matrix multiply(const Matrix &other);
    Matrix addition(const Matrix &other);

    Matrix transpose();
    Matrix multiply_scalar(float scalar);
    Matrix gelu();
};
