#pragma once

#include <vector>
#include "safetensors.hpp"
#include "matrix.hpp"
#include "embedding.hpp"
#include "transformer_block.hpp"

class GPT
{
public:
    GPT(SafeTensors &weights, int n_layers = 12, int heads = 12);

    Matrix forward(const std::vector<int> &token_ids);
    int next_token(const std::vector<int> &token_ids);                  // argmax  or whatever temperature sampling u want of last row of forward
    std::vector<int> generate(std::vector<int> token_ids, int max_new); // greedy loop
    const int EOT_TOKEN = 50256;
    int max_context() const { return MAX_POSITIONS; }

private:
    Embedding embedding;
    std::vector<TransformerBlock> blocks;

    Matrix LN_F_WEIGHT; // final layernorm gamma
    Matrix LN_F_BIAS;   // final layernorm beta
    Matrix WTE_T;       // wte transposed, [dmodel, vocab]. output head is tied to input embedding
    // this bascially converts the output dmodel back into the vocab size . its the input just transposed
    int MAX_POSITIONS; // rows of wpe
};
