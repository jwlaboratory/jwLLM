#include "tensor.hpp"

#include <stdexcept>

using std::vector;

Tensor::Tensor(vector<int> dims, vector<float> passed_data)
{

    size_t max_total = dims[0];
    for (size_t i = 1; i < dims.size(); i++)
    {
        max_total *= dims[i];
    }
    if (max_total != passed_data.size())
    {
        throw std::invalid_argument("dimensions must match total amount of data");
    }

    dimension_sizes = dims;
    data = passed_data;
}

Tensor Tensor::multiply(Tensor other)
{

    // x,y,z @ x,y,z
    // x must match
    // z1 must match y2

    // output dims x, y1, z2
    Tensor output;

    // check dims
    if (dimension_sizes.size() == 3)
    {
        if (other.dimension_sizes.size() != 3 || other.dimension_sizes[0] != dimension_sizes[0])
        {
            throw std::invalid_argument("dims dont match");
        }

        if (dimension_sizes[2] != other.dimension_sizes[1])
        {
            throw std::invalid_argument("dims dont match");
        }

        output = new Tensor(new vector<3, dimension_sizes[1], other.dimension_sizes[2]>, malloc(size_of(float) * 3 * dimension_sizes[1] * other.dimension_sizes[2]));
    }
    // must be 2d
    else
    {
        if (dimension_sizes[1] != other.dimension_sizes[0])
        {
            throw std::invalid_argument("dims dont match");
        }

        output = new Tensor(new vector<2, dimension_sizes[1], other.dimension_sizes[0]>, malloc(size_of(float) * 3 * dimension_sizes[1] * other.dimension_sizes[0]));
    }

    // do actual multiplication

    return other;
}
