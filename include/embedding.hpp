#pragma once

#include <vector>
#include "matrix.hpp"
#include "safetensors.hpp"

class Embedding
{
public:
    Embedding(SafeTensors &weights);
    Matrix tokenized_to_embed(const std::vector<int> &token_ids);
    void apply_positional_encoding(Matrix &token_embeddings);

private:
    Matrix WORD_TOKEN_EMBEDDING;
    Matrix WORD_POSITIONAL_EMBEDDING;
};
