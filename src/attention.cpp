#include "attention.hpp"
#include "matrix.hpp"

Attention::Attention(SafeTensors &weights, const std::string &prefix, int heads)
{
    ATTENTION_WEIGHTS = weights.get(prefix + "attn.c_attn.weight");
    ATTENTION_BIAS = weights.get(prefix + "attn.c_attn.bias");
}

Matrix Attention::forward(const Matrix &x)
{
    // lets create k, q, v
    // fused multiplication intuition

    // x= 3x10 (3 seq len, dmodel=10)
    // fused = 10x10 but 3 stacked HORiZONTALLY, so its 10x30

    // output 10x30
    // Q = Y[:, 0:10] K = Y[:, 10:20] V = Y[:, 20:30]
    int dmodel = ATTENTION_WEIGHTS.rows;
    Matrix fusedKQV = (x.multiply(ATTENTION_WEIGHTS)).broadcast_add_row(ATTENTION_BIAS);

    Matrix K = fusedKQV.slice_cols(0, dmodel);
    Matrix Q = fusedKQV.slice_cols(dmodel, dmodel);
    Matrix V = fusedKQV.slice_cols(dmodel * 2, dmodel);
}