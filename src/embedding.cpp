
#include "matrix.hpp"
#include "embedding.hpp"
using namespace std;

Embedding::Embedding(SafeTensors &weights)
{
    WORD_TOKEN_EMBEDDING = weights.get("wte.weight");
    WORD_POSITIONAL_EMBEDDING = weights.get("wpe.weight");
}

Matrix Embedding::tokenized_to_embed(const Matrix &token_ids)
{
    // token_ids is a single sequence: 1 row, seq_len cols, ids stored as floats.
    int d_model = WORD_TOKEN_EMBEDDING.cols;
    int seq_len = token_ids.cols;
    std::vector<float> out(seq_len * d_model);

    for (int i = 0; i < seq_len; i++)
    {
        int token_id = static_cast<int>(token_ids.data[i]);

        for (int g = 0; g < d_model; g++)
        {
            out[i * d_model + g] = WORD_TOKEN_EMBEDDING.data[token_id * d_model + g];
            // out is flat array, word_token embedding is also flat array
        }
    }

    return Matrix(seq_len, d_model, out);
}

void Embedding::apply_positional_encoding(Matrix &token_embeddings)
{
    // seq len rows
    // dmodel cols

    // WORD_POSITIONAL_EMBEDDING is a table of size: position rows, dmodel cols

    int seq_len = token_embeddings.rows;
    int dmodel = token_embeddings.cols;

    for (int i = 0; i < seq_len; i++)
    {
        for (int g = 0; g < token_embeddings.cols; g++)
        {
            token_embeddings.data[i * dmodel + g] += WORD_POSITIONAL_EMBEDDING.data[i * dmodel + g];
        }
    }
}
