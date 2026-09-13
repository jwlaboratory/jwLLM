#include "matrix.hpp"
#include <cmath>
#include <stdexcept>

using std::vector;

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows, int cols, vector<float> passed_data)
{

    size_t max_total = rows * cols;
    if (max_total != passed_data.size())
        throw std::invalid_argument("dimensions must match total amount of data");

    this->rows = rows;
    this->cols = cols;
    this->data = passed_data;
}

Matrix Matrix::multiply(const Matrix &other)
{
    // check dims
    if (cols != other.rows)
        throw std::invalid_argument("dims dont match");

    // do actual multiplication
    std::vector<float> out(rows * other.cols);

    for (int i = 0; i < this->rows; i++)
    {
        // for each row

        for (int q = 0; q < other.cols; q++)
        {
            // each col of the other now
            float sum = 0;
            for (int g = 0; g < this->cols; g++)
            {
                sum += this->data[i * this->cols + g] * other.data[g * other.cols + q];
            }

            out[i * other.cols + q] = sum;
        }
    }

    return Matrix(rows, other.cols, out);
}

Matrix Matrix::addition(const Matrix &other)
{
    // check dims
    if (cols != other.cols || rows != other.rows)
        throw std::invalid_argument("dims dont match");

    // do actual addition
    std::vector<float> out(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        out[i] = this->data[i] + other.data[i];
    }

    return Matrix(rows, cols, out);
}

Matrix Matrix::transpose()
{
    // we store row x col
    // we need to swap to col x row

    // ie transpose:
    // [a,b,c,d,e,f]
    // 2x3 --> 3x2
    //[a, d, b, e, c, f]

    // 0,1 --> 1,0
    //

    vector<float> out(this->data.size());
    for (int i = 0; i < this->rows; i++)
    {
        for (int g = 0; g < this->cols; g++)
        {
            int newRow = g;
            int newCol = i;
            int total_per_row_new = this->rows;

            out[newRow * total_per_row_new + newCol] = this->data[i * this->cols + g];
        }
    }
    return Matrix(this->cols, this->rows, out);
}

Matrix Matrix::multiply_scalar(float scalar)
{
    vector<float> out(this->data.size());
    for (int i = 0; i < this->data.size(); i++)
    {
        out[i] = this->data[i] * scalar;
    }
    return Matrix(this->rows, this->cols, out);
}

Matrix Matrix::gelu()
{
    vector<float> out(this->data.size());
    for (int i = 0; i < this->data.size(); i++)
    {
        float x = this->data[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
    return Matrix(this->rows, this->cols, out);
}

Matrix Matrix::broadcast_add_row(const Matrix &row)
{
    // we have a matrix [seqlen x dmodel]
    // we want to add a bias of size [dmodel]
    // we broadcast so this adds to each row

    if (row.rows != 1 || this->cols != row.cols)
    {
        throw std::invalid_argument("to broadcast add, must be row size =1");
    }

    vector<float> out(this->data.size());

    for (int i = 0; i < this->rows; i++)
    {
        for (int g = 0; g < this->cols; g++)
        {
            out[i * this->cols + g] = this->data[i * this->cols + g] + row.data[g];
        }
    }
    return Matrix(this->rows, this->cols, out);
}

Matrix Matrix::broadcast_multiply_row(const Matrix &row)
{
    if (row.rows != 1 || this->cols != row.cols)
    {
        throw std::invalid_argument("to broadcast add, must be row size =1");
    }

    vector<float> out(this->data.size());

    for (int i = 0; i < this->rows; i++)
    {
        for (int g = 0; g < this->cols; g++)
        {
            out[i * this->cols + g] = this->data[i * this->cols + g] * row.data[g];
        }
    }
    return Matrix(this->rows, this->cols, out);
}

Matrix Matrix::softmax_rows()
{
    throw std::logic_error("not implemented");
}

Matrix Matrix::slice_cols(int start, int len)
{
    throw std::logic_error("not implemented");
}

Matrix Matrix::concat_cols(const Matrix &other)
{
    throw std::logic_error("not implemented");
}
