#pragma once
#include "safetensors.hpp"
#include "matrix.hpp"

class Attention
{
public:
    Attention(SafeTensors &weights, const std::string &prefix, int heads);
    Matrix forward(const Matrix &x);

private:
    Matrix ATTENTION_WEIGHTS;
    Matrix ATTENTION_BIAS;
    Matrix PROJECTION_WEIGHTS;
    Matrix PROJECTION_BIAS;
    int heads;
    int dmodel;
};
