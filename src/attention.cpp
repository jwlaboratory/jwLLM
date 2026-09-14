#include "attention.hpp"
#include "matrix.hpp"

Attention::Attention(SafeTensors &weights, const std::string &prefix, int heads)
{
    ATTENTION_WEIGHTS = weights.get(prefix + "attn.c_attn.weight");
    ATTENTION_BIAS = weights.get(prefix + "attn.c_attn.bias");

    // projection s after attention to mix the splitting
    PROJECTION_WEIGHTS = weights.get(prefix + "attn.c_proj.weight");
    PROJECTION_BIAS = weights.get(prefix + "attn.c_proj.bias");

    this->heads = heads;
    dmodel = ATTENTION_WEIGHTS.rows;

    if (dmodel % heads != 0)
    {
        throw std::invalid_argument("dmodel must be cleanly divisble by dhead and dmodel");
    }
}

Matrix Attention::forward(const Matrix &x)
{
    // lets create k, q, v
    // fused multiplication intuition

    // x= 3x10 (3 seq len, dmodel=10)
    // fused = 10x10 but 3 stacked HORiZONTALLY, so its 10x30

    // output 10x30
    // Q = Y[:, 0:10] K = Y[:, 10:20] V = Y[:, 20:30]
    Matrix FusedQKV = (x.multiply(ATTENTION_WEIGHTS)).broadcast_add_row(ATTENTION_BIAS);

    Matrix Q = FusedQKV.slice_cols(0, dmodel);
    Matrix K = FusedQKV.slice_cols(dmodel, dmodel);
    Matrix V = FusedQKV.slice_cols(dmodel * 2, dmodel);
    // shape is [seqlen, d_model]

    // now, we need to do spliting
    int d_head = dmodel / heads;

    // say we have 4 heads and dmodel of 16
    // 0-4, 4-8, 8-12, 12-16
    // 0,   1,   2,    3
    Matrix output;
    for (int h = 0; h < heads; h++)
    {
        Matrix q_head = Q.slice_cols(h * d_head, d_head);
        Matrix k_head = K.slice_cols(h * d_head, d_head);
        Matrix v_head = V.slice_cols(h * d_head, d_head);
        // shape is [seq, dhead]

        // transpose k
        Matrix k_t = k_head.transpose(); //[dhead, seq]

        Matrix Q_kt = q_head.multiply(k_t); //[seq, seq]
        // tells us how much each entry in x relates to x, in that zone of embedding focus

        // divide by the sqrt d_k
        Matrix pre_mask = Q_kt.multiply_scalar(1 / std::sqrt(d_head));
        Matrix masked = pre_mask.mask_causal();
        Matrix softmaxed = masked.softmax_rows(); // still [seq, seq]

        Matrix post_v = softmaxed.multiply(v_head);
        // [seq,seq] * [seq, dhead]
        // now back to [seq, dhead]

        // append
        if (h == 0)
            output = post_v;

        else
            output = output.concat_cols(post_v);
    }

    // do the projections
    return output.multiply(PROJECTION_WEIGHTS).broadcast_add_row(PROJECTION_BIAS);
    //[seq, dmodel]
}