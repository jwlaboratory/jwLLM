#pragma once

#include <vector>
#include "matrix.hpp"

class Embedding
{
public:
    Embedding(std::string file_path_safe_tensors);
    Matrix tokenized_to_embed(const std::vector<int> &token_ids);
    Matrix apply_positional_encoding(Matrix token_embeddings);

private:
    Matrix WORD_TOKEN_EMBEDDING;
    Matrix WORD_POSITIONAL_EMBEDDING;
};
