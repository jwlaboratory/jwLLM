#include "gpt.hpp"
#include <stdexcept>
#include <string>

GPT::GPT(SafeTensors &weights, int n_layers, int heads)
    : embedding(weights)
{
    for (int i = 0; i < n_layers; i++)
    {
        blocks.push_back(TransformerBlock(weights, "h." + std::to_string(i) + ".", heads));
    }

    LN_F_WEIGHT = weights.get("ln_f.weight");
    LN_F_BIAS = weights.get("ln_f.bias");

    WTE_T = weights.get("wte.weight").transpose(); // this is the initial vocab -> dmodel transposed so we can go backward
    MAX_POSITIONS = weights.get("wpe.weight").rows;
}

Matrix GPT::forward(const std::vector<int> &token_ids)
{
    if (token_ids.empty())
    {
        throw std::invalid_argument("forward needs at least one token");
    }

    // [seq, dmodel]
    Matrix x = embedding.tokenized_to_embed(token_ids);
    embedding.apply_positional_encoding(x);

    for (TransformerBlock &block : blocks)
    {
        x = block.forward(x);
    }

    x = x.layernorm(LN_F_WEIGHT, LN_F_BIAS);
    return x.multiply(WTE_T); // back to vocab size
}

int GPT::next_token(const std::vector<int> &token_ids)
{
    Matrix logits = forward(token_ids);

    // arg max for now
    int last = (logits.rows - 1) * logits.cols;
    int best = 0;
    for (int j = 1; j < logits.cols; j++)
    {
        if (logits.data[last + j] > logits.data[last + best])
        {
            best = j;
        }
    }
    return best;
}

std::vector<int> GPT::generate(std::vector<int> token_ids, int max_new)
{
    for (int i = 0; i < max_new; i++)
    {
        if ((int)token_ids.size() >= MAX_POSITIONS)
        {
            break;
        }
        int next = next_token(token_ids);
        if (next == EOT_TOKEN)
        {
            break;
        }
        token_ids.push_back(next);
    }
    return token_ids;
}

// Tensor	Shape	What it is
// ln_1.weight	768	Layernorm gamma, before attention
// ln_1.bias	768	Layernorm beta, before attention
// attn.c_attn.weight	768 x 2304	Fused query, key, value projection
// attn.c_attn.bias	2304	Bias for the above
// attn.c_proj.weight	768 x 768	Output projection after head concat
// attn.c_proj.bias	768	Bias for the above
// ln_2.weight	768	Layernorm gamma, before the MLP
// ln_2.bias	768	Layernorm beta, before the MLP
// mlp.c_fc.weight	768 x 3072	Expand to 4x width
// mlp.c_fc.bias	3072	Bias for the above
// mlp.c_proj.weight	3072 x 768	Contract back to 768
// mlp.c_proj.bias	768	Bias for the above
// attn.bias	1 x 1 x 1024 x 1024	Not a bias. Ignore it.
