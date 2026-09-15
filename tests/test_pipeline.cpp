#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "embedding.hpp"
#include "matrix.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"

// End-to-end test of the pipeline wired up in main.cpp: raw input ->
// tokenize -> embed -> positional encode, against the real model files.
TEST(Pipeline, TokenizeEmbedAndPositionallyEncode)
{
    Tokenizer tok("data/vocab.json", "data/merges.txt");
    SafeTensors weights("data/model.safetensors");
    Embedding embedding(weights);

    std::string input = "Hello world!!!";

    // 1) input -> 2) tokenize
    std::vector<int> token_ids = tok.encode(input);
    ASSERT_GT(token_ids.size(), 0u);

    // decoding what we tokenized should give back the original input
    ASSERT_EQ(tok.decode(token_ids), input);

    // 3) embed
    Matrix embedded = embedding.tokenized_to_embed(token_ids);
    ASSERT_EQ(embedded.rows, static_cast<int>(token_ids.size()));
    ASSERT_GT(embedded.cols, 0);

    std::vector<float> pre_positional = embedded.data;

    // 4) positional encode
    embedding.apply_positional_encoding(embedded);

    ASSERT_EQ(embedded.rows, static_cast<int>(token_ids.size()));
    ASSERT_EQ(embedded.cols, static_cast<int>(pre_positional.size() / token_ids.size()));

    // positional encoding must add to the token embedding, not discard it:
    // the result should differ from the pre-positional embedding...
    EXPECT_NE(embedded.data, pre_positional);

    // ...but two tokens at different positions should no longer share
    // identical rows purely by coincidence of positional overwrite, and if
    // the same token id repeats at two positions its rows must now differ.
    if (token_ids.size() > 1 && token_ids[0] == token_ids[1])
    {
        int d_model = embedded.cols;
        std::vector<float> row0(embedded.data.begin(), embedded.data.begin() + d_model);
        std::vector<float> row1(embedded.data.begin() + d_model, embedded.data.begin() + 2 * d_model);
        EXPECT_NE(row0, row1);
    }
}
