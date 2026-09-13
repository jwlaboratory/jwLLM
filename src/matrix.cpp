#include "matrix.hpp"

#include <stdexcept>

using std::vector;

Matrix::Matrix(int rows, int cols, vector<float> passed_data)
{

    size_t max_total = rows * cols;
    if (max_total != passed_data.size())
        throw std::invalid_argument("dimensions must match total amount of data");

    rows = rows;
    cols = cols;
    data = passed_data;
}

Matrix Matrix::multiply(Matrix other)
{
    Matrix output;

    // check dims
    if (cols != other.rows)
        throw std::invalid_argument("dims dont match");

    // do actual multiplication
}
