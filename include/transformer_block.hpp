#pragma once
#include "safetensors.hpp"
#include "matrix.hpp"
#include "attention.hpp"
#include "mlp.hpp"

// layernorm -> attention --> residual,
// layernorm-> mlp -> residual

class TransformerBlock
{
public:
    TransformerBlock(SafeTensors &weights, const std::string &prefix, int heads);
    Matrix forward(const Matrix &x);

private:
    Attention attn;
    MLP mlp;

    Matrix LN1_WEIGHT; // layer norm gamma BEFORE transformer
    Matrix LN1_BIAS;   // laryne norm beta BEFORE

    Matrix LN2_WEIGHT; // gamma AFTER (before MLP)
    Matrix LN2_BIAS;   // beta
};
