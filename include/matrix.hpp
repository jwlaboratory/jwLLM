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
    Matrix multiply(const Matrix &other) const;
    Matrix addition(const Matrix &other) const; // we need the functions returning cost so we can chain together multiple functions

    Matrix transpose() const;
    Matrix multiply_scalar(float scalar) const;
    Matrix gelu() const;

    Matrix broadcast_add_row(const Matrix &row) const;
    Matrix broadcast_multiply_row(const Matrix &row) const;
    Matrix mask_causal(float big_negative = -1e9f) const;
    Matrix softmax_rows() const;
    Matrix layernorm(const Matrix &gamma, const Matrix &beta, float eps = 1e-5f) const;
    Matrix slice_cols(int start, int len) const;
    Matrix concat_cols(const Matrix &other) const;
};
