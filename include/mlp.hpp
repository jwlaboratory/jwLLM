#pragma once
#include "safetensors.hpp"
#include "matrix.hpp"

// feed forward part of attention block
class MLP
{
public:
    MLP(SafeTensors &weights, const std::string &prefix);
    Matrix forward(const Matrix &x) const;

private:
    Matrix FC_WEIGHTS;
    Matrix FC_BIAS;
    Matrix PROJECTION_WEIGHTS;
    Matrix PROJECTION_BIAS;
};
