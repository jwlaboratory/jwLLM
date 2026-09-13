#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "embedding.hpp"
#include "matrix.hpp"

using json = nlohmann::json;

namespace
{
    // wte: 4 tokens x 3 dims, wpe: 5 positions x 3 dims.
    const std::vector<float> kWte = {
        1.0f, 2.0f, 3.0f,
        10.0f, 20.0f, 30.0f,
        100.0f, 200.0f, 300.0f,
        1000.0f, 2000.0f, 3000.0f};

    const std::vector<float> kWpe = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f};

    // Writes a minimal safetensors file containing transformer.wte.weight
    // and transformer.wpe.weight, matching the layout Embedding's
    // constructor expects, and returns its path.
    std::string write_fixture_safetensors()
    {
        std::string path = "build/test_embedding_fixture.safetensors";

        uint64_t wte_bytes = kWte.size() * sizeof(float);
        uint64_t wpe_bytes = kWpe.size() * sizeof(float);

        json header = {
            {"transformer.wte.weight", {{"dtype", "F32"}, {"shape", {4, 3}}, {"data_offsets", {0, wte_bytes}}}},
            {"transformer.wpe.weight", {{"dtype", "F32"}, {"shape", {5, 3}}, {"data_offsets", {wte_bytes, wte_bytes + wpe_bytes}}}},
        };
        std::string header_str = header.dump();
        uint64_t header_len = header_str.size();

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
        out.write(header_str.data(), header_str.size());
        out.write(reinterpret_cast<const char *>(kWte.data()), wte_bytes);
        out.write(reinterpret_cast<const char *>(kWpe.data()), wpe_bytes);

        return path;
    }

    Embedding make_embedding()
    {
        return Embedding(write_fixture_safetensors());
    }
} // namespace

TEST(Embedding, ThrowsOnMissingFile)
{
    EXPECT_THROW(Embedding("data/does_not_exist.safetensors"), std::runtime_error);
}

TEST(Embedding, TokenizedToEmbedLooksUpCorrectRows)
{
    Embedding emb = make_embedding();

    Matrix result = emb.tokenized_to_embed({2, 0, 3});

    EXPECT_EQ(result.rows, 3);
    EXPECT_EQ(result.cols, 3);
    EXPECT_EQ(result.data, std::vector<float>({100.0f, 200.0f, 300.0f,
                                               1.0f, 2.0f, 3.0f,
                                               1000.0f, 2000.0f, 3000.0f}));
}

TEST(Embedding, ApplyPositionalEncodingAddsRatherThanOverwrites)
{
    Embedding emb = make_embedding();

    Matrix embeddings = emb.tokenized_to_embed({1, 2});
    emb.apply_positional_encoding(embeddings);

    // token embedding for id 1 is row1 of wte, plus wpe row0 (position 0)
    EXPECT_FLOAT_EQ(embeddings.data[0], 10.0f + 0.1f);
    EXPECT_FLOAT_EQ(embeddings.data[1], 20.0f + 0.2f);
    EXPECT_FLOAT_EQ(embeddings.data[2], 30.0f + 0.3f);

    // token embedding for id 2 is row2 of wte, plus wpe row1 (position 1)
    EXPECT_FLOAT_EQ(embeddings.data[3], 100.0f + 0.4f);
    EXPECT_FLOAT_EQ(embeddings.data[4], 200.0f + 0.5f);
    EXPECT_FLOAT_EQ(embeddings.data[5], 300.0f + 0.6f);
}

TEST(Embedding, ApplyPositionalEncodingKeepsDifferentTokensDistinguishable)
{
    Embedding emb = make_embedding();

    // Two different tokens at the same position (0) must not collapse to
    // the same vector after positional encoding is applied.
    Matrix a = emb.tokenized_to_embed({0});
    Matrix b = emb.tokenized_to_embed({3});

    emb.apply_positional_encoding(a);
    emb.apply_positional_encoding(b);

    EXPECT_NE(a.data, b.data);
}
