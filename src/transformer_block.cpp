#include "transformer_block.hpp"
#include <stdexcept>

TransformerBlock::TransformerBlock(SafeTensors &weights, const std::string &prefix, int heads)
    : attn(weights, prefix, heads), mlp(weights, prefix)
{

    LN1_WEIGHT = weights.get(prefix + "ln_1.weight");
    LN1_BIAS = weights.get(prefix + "ln_1.bias");
    LN2_WEIGHT = weights.get(prefix + "ln_2.weight");
    LN2_BIAS = weights.get(prefix + "ln_2.bias");
}

Matrix TransformerBlock::forward(const Matrix &x)
{

    // input = [seq, dmodel]
    // output = [seq, dmodel]

    // do layernorm and attention. y = attention(layermorm(x))
    // z = y + x
    // do layernorm and mlp
    // out = MLP(layernorm(z)) +  z

    Matrix attention_plus_residual = attn.forward(x.layernorm(LN1_WEIGHT, LN1_BIAS)).addition(x);
    Matrix MLP_plus_residual = mlp.forward(attention_plus_residual.layernorm(LN2_WEIGHT, LN2_BIAS)).addition(attention_plus_residual);

    return MLP_plus_residual;
}
