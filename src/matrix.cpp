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

    vector<float> out(this->data.size());

    // loop 0: for each row
    for (int row = 0; row < this->rows; row++)
    {

        // loop one: find max val in the row
        float max = this->data[row * this->cols + 0];
        for (int g = 0; g < this->cols; g++)
        {
            if (this->data[row * this->cols + g] > max)
            {
                max = this->data[row * this->cols + g];
            }
        }

        // loop 2: calc sum and set each index to the e^(si-max)
        float sum_of_all = 0;
        for (int g = 0; g < this->cols; g++)
        {
            out[row * this->cols + g] = std::exp(this->data[row * this->cols + g] - max);
            sum_of_all += out[row * this->cols + g];
        }

        // loop 3: divide all by sum
        for (int g = 0; g < this->cols; g++)
        {
            out[row * this->cols + g] /= sum_of_all;
        }
    }

    return Matrix(this->rows, this->cols, out);
}

// given a matrix, give the data in col start, start+1, start+2... start+len
Matrix Matrix::slice_cols(int start, int len)
{
    if (start < 0 || len <= 0 || start + len > this->cols)
        throw std::invalid_argument("slice out of range");

    vector<float> out(this->rows * len);
    int index = 0;
    for (int i = 0; i < this->rows; i++)
    {
        for (int g = 0; g < this->cols; g++)
        {
            if (start <= g && g < start + len)
            {
                out[index] = this->data[i * this->cols + g];
                index++;
            }
        }
    }
    return Matrix(this->rows, len, out);
}

Matrix Matrix::concat_cols(const Matrix &other)
{

    if (other.rows != this->rows)
    {
        throw std::invalid_argument("got to have same rows for concat");
    }

    vector<float> out(this->data.size() + other.data.size());
    int max_len = this->cols + other.cols;

    for (int i = 0; i < this->rows; i++)
    {
        for (int g = 0; g < max_len; g++)
        {
            if (g < this->cols)
            {
                out[i * max_len + g] = this->data[i * this->cols + g];
            }
            else
            {
                out[i * max_len + g] = other.data[i * other.cols + g - this->cols];
            }
        }
    }
    return Matrix(this->rows, max_len, out);
}
// normalize each row to mean 0 / variance 1, then scale by gamma and shift by beta
Matrix Matrix::layernorm(const Matrix &gamma, const Matrix &beta, float eps)
{

    vector<float> out(this->data.size());

    for (int i = 0; i < this->rows; i++)
    {

        // for each row
        float mean = 0;
        for (int g = 0; g < this->cols; g++)
        {
            mean += this->data[i * this->cols + g];
        }
        mean /= this->cols;

        float var = 0;
        for (int g = 0; g < this->cols; g++)
        {
            float diff = this->data[i * this->cols + g] - mean;
            var += diff * diff;
        }
        var /= this->cols;
        // vaiance is difference^2) averaged out
        // variance is sigma (standard deviation squared)

        // z sciore - (x-u)/sigma
        // we basically calcualting this

        // eps to pevent divide by zero

        for (int g = 0; g < this->cols; g++)
        {
            out[i * this->cols + g] = (this->data[i * this->cols + g] - mean) / std::sqrt(var + eps);
        }
    }

    // beta and gamma are learned so model decides what to expand. gamma is sscaler, beta is additive. //eps is to prevent divide by zero
    Matrix out_m = Matrix(this->rows, this->cols, out);
    return (out_m.broadcast_multiply_row(gamma)).broadcast_add_row(beta);
}
