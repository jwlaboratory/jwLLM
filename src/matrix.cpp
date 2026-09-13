#include "matrix.hpp"

#include <stdexcept>

using std::vector;

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
