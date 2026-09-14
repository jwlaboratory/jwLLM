#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "attention.hpp"
#include "matrix.hpp"
#include "safetensors.hpp"

using json = nlohmann::json;

namespace
{
    // A dmodel x dmodel identity block, scaled.
    std::vector<float> eye(int dmodel, float scale = 1.0f)
    {
        std::vector<float> out(dmodel * dmodel, 0.0f);
        for (int i = 0; i < dmodel; i++)
        {
            out[i * dmodel + i] = scale;
        }
        return out;
    }

    std::vector<float> zeros(int dmodel)
    {
        return std::vector<float>(dmodel * dmodel, 0.0f);
    }

    // GPT-2 stores one fused c_attn weight of shape [dmodel, 3*dmodel]: the
    // Q, K and V projections stacked horizontally in that order. Builds that
    // from three separate dmodel x dmodel blocks.
    std::vector<float> fuse_qkv(int dmodel,
                                const std::vector<float> &wq,
                                const std::vector<float> &wk,
                                const std::vector<float> &wv)
    {
        int fused_cols = 3 * dmodel;
        std::vector<float> out(dmodel * fused_cols, 0.0f);
        for (int r = 0; r < dmodel; r++)
        {
            for (int c = 0; c < dmodel; c++)
            {
                out[r * fused_cols + c] = wq[r * dmodel + c];
                out[r * fused_cols + dmodel + c] = wk[r * dmodel + c];
                out[r * fused_cols + 2 * dmodel + c] = wv[r * dmodel + c];
            }
        }
        return out;
    }

    // Writes a minimal safetensors file holding just the tensors Attention's
    // constructor asks for, under the prefix "h.0.".
    std::string write_fixture_safetensors(const std::string &path,
                                          int dmodel,
                                          const std::vector<float> &fused_weight,
                                          const std::vector<float> &fused_bias)
    {
        uint64_t weight_bytes = fused_weight.size() * sizeof(float);
        uint64_t bias_bytes = fused_bias.size() * sizeof(float);

        json header = {
            {"h.0.attn.c_attn.weight", {{"dtype", "F32"}, {"shape", {dmodel, 3 * dmodel}}, {"data_offsets", {0, weight_bytes}}}},
            {"h.0.attn.c_attn.bias", {{"dtype", "F32"}, {"shape", {3 * dmodel}}, {"data_offsets", {weight_bytes, weight_bytes + bias_bytes}}}},
        };
        std::string header_str = header.dump();
        uint64_t header_len = header_str.size();

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
        out.write(header_str.data(), header_str.size());
        out.write(reinterpret_cast<const char *>(fused_weight.data()), weight_bytes);
        out.write(reinterpret_cast<const char *>(fused_bias.data()), bias_bytes);

        return path;
    }

    Attention make_attention(const std::string &path,
                             int dmodel,
                             int heads,
                             const std::vector<float> &wq,
                             const std::vector<float> &wk,
                             const std::vector<float> &wv,
                             const std::vector<float> &bias = {})
    {
        std::vector<float> fused_bias = bias.empty()
                                            ? std::vector<float>(3 * dmodel, 0.0f)
                                            : bias;

        SafeTensors weights(write_fixture_safetensors(
            path, dmodel, fuse_qkv(dmodel, wq, wk, wv), fused_bias));

        return Attention(weights, "h.0.", heads);
    }

    // Q and K projections zeroed, V left as identity. Every score is then 0,
    // so after causal masking row i attends *uniformly* to rows 0..i and the
    // output is the running mean of the input rows. Makes the mask, the
    // softmax and the weighted sum all checkable by hand.
    Attention make_uniform_attention(const std::string &path,
                                     int dmodel,
                                     int heads,
                                     const std::vector<float> &bias = {})
    {
        return make_attention(path, dmodel, heads,
                              zeros(dmodel), zeros(dmodel), eye(dmodel), bias);
    }
} // namespace

TEST(Attention, OutputIsSeqLenByDModel)
{
    Attention attn = make_uniform_attention("build/test_attention_shape.safetensors", 4, 2);

    Matrix out = attn.forward(Matrix(3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                            5.0f, 6.0f, 7.0f, 8.0f,
                                            9.0f, 10.0f, 11.0f, 12.0f}));

    // Heads are concatenated back together, so the width is dmodel, not
    // 3*dmodel and not dmodel*heads.
    EXPECT_EQ(out.rows, 3);
    EXPECT_EQ(out.cols, 4);
}

TEST(Attention, SingleTokenAttendsOnlyToItself)
{
    // With one token the softmax is over a single score, so it is exactly
    // 1.0 whatever the scores are, and the output is just that token's V row.
    Attention attn = make_attention("build/test_attention_single.safetensors", 2, 1,
                                    eye(2), eye(2), eye(2, 2.0f));

    Matrix out = attn.forward(Matrix(1, 2, {3.0f, 4.0f}));

    ASSERT_EQ(out.rows, 1);
    ASSERT_EQ(out.cols, 2);
    EXPECT_FLOAT_EQ(out.data[0], 6.0f);
    EXPECT_FLOAT_EQ(out.data[1], 8.0f);
}

TEST(Attention, UniformScoresAverageOverPastTokensOnly)
{
    Attention attn = make_uniform_attention("build/test_attention_uniform.safetensors", 2, 1);

    Matrix out = attn.forward(Matrix(3, 2, {1.0f, 2.0f,
                                            3.0f, 4.0f,
                                            5.0f, 6.0f}));

    ASSERT_EQ(out.rows, 3);
    ASSERT_EQ(out.cols, 2);

    // row 0 sees only itself
    EXPECT_FLOAT_EQ(out.data[0], 1.0f);
    EXPECT_FLOAT_EQ(out.data[1], 2.0f);

    // row 1 is the mean of rows 0 and 1
    EXPECT_FLOAT_EQ(out.data[2], 2.0f);
    EXPECT_FLOAT_EQ(out.data[3], 3.0f);

    // row 2 is the mean of all three. If the causal mask were missing, row 0
    // and row 1 above would be this same value.
    EXPECT_FLOAT_EQ(out.data[4], 3.0f);
    EXPECT_FLOAT_EQ(out.data[5], 4.0f);
}

TEST(Attention, FutureTokensDoNotChangeEarlierRows)
{
    // The causality invariant, with ordinary weights rather than a rigged
    // uniform setup: changing the last token must leave every earlier output
    // row bit-identical.
    std::vector<float> wq = {0.5f, -1.0f, 2.0f, 0.25f};
    std::vector<float> wk = {1.5f, 0.75f, -0.5f, 1.0f};
    std::vector<float> wv = {-2.0f, 1.0f, 0.5f, 3.0f};

    Attention attn = make_attention("build/test_attention_causal.safetensors", 2, 1,
                                    wq, wk, wv);

    Matrix a = attn.forward(Matrix(3, 2, {1.0f, 2.0f,
                                          3.0f, 4.0f,
                                          5.0f, 6.0f}));

    Matrix b = attn.forward(Matrix(3, 2, {1.0f, 2.0f,
                                          3.0f, 4.0f,
                                          -100.0f, 70.0f}));

    ASSERT_EQ(a.cols, 2);
    ASSERT_EQ(b.cols, 2);
    for (int i = 0; i < 4; i++)
    {
        EXPECT_FLOAT_EQ(a.data[i], b.data[i]) << "row " << (i / 2) << " leaked future information";
    }
}

TEST(Attention, ScalesScoresByInverseSqrtHeadDim)
{
    // dmodel 2, one head, identity projections, so Q = K = V = x.
    Attention attn = make_attention("build/test_attention_scale.safetensors", 2, 1,
                                    eye(2), eye(2), eye(2));

    Matrix out = attn.forward(Matrix(2, 2, {1.0f, 0.0f,
                                            0.0f, 1.0f}));

    // Row 1's raw scores against rows 0 and 1 are [0, 1]. Scaled by
    // 1/sqrt(d_head) they become [0, 1/sqrt(2)]. Dropping the scaling gives
    // weights of [0.269, 0.731] instead of the [0.330, 0.670] below.
    float scaled = 1.0f / std::sqrt(2.0f);
    float e = std::exp(scaled);
    float w_past = 1.0f / (1.0f + e);
    float w_self = e / (1.0f + e);

    ASSERT_EQ(out.rows, 2);
    EXPECT_FLOAT_EQ(out.data[0], 1.0f);
    EXPECT_FLOAT_EQ(out.data[1], 0.0f);
    EXPECT_FLOAT_EQ(out.data[2], w_past);
    EXPECT_FLOAT_EQ(out.data[3], w_self);
}

TEST(Attention, AddsTheValueSliceOfTheFusedBias)
{
    // Bias layout mirrors the weight: [Q bias | K bias | V bias]. Only the V
    // third is given a value, so it lands directly on the output. Reading the
    // bias at the wrong offset picks up zeros and this fails.
    std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 20.0f};

    Attention attn = make_uniform_attention("build/test_attention_bias.safetensors", 2, 1, bias);

    Matrix out = attn.forward(Matrix(2, 2, {1.0f, 2.0f,
                                            3.0f, 4.0f}));

    // V rows are the inputs shifted by the bias, then averaged as before.
    ASSERT_EQ(out.rows, 2);
    EXPECT_FLOAT_EQ(out.data[0], 11.0f);
    EXPECT_FLOAT_EQ(out.data[1], 22.0f);
    EXPECT_FLOAT_EQ(out.data[2], 12.0f);
    EXPECT_FLOAT_EQ(out.data[3], 23.0f);
}

TEST(Attention, HeadsAttendOverTheirOwnColumnsOnly)
{
    // dmodel 4 split into two heads of width 2. Head 0 sees columns 0-1 and
    // head 1 sees columns 2-3, so changing only the second half of the input
    // must leave the first half of the output untouched. With a single head
    // spanning all four columns this would not hold.
    Attention attn = make_attention("build/test_attention_heads.safetensors", 4, 2,
                                    eye(4), eye(4), eye(4));

    Matrix a = attn.forward(Matrix(2, 4, {1.0f, 0.0f, 5.0f, 0.0f,
                                          0.0f, 1.0f, 0.0f, 5.0f}));

    Matrix b = attn.forward(Matrix(2, 4, {1.0f, 0.0f, 9.0f, 9.0f,
                                          0.0f, 1.0f, 9.0f, 9.0f}));

    ASSERT_EQ(a.cols, 4);
    ASSERT_EQ(b.cols, 4);
    for (int row = 0; row < 2; row++)
    {
        EXPECT_FLOAT_EQ(a.data[row * 4 + 0], b.data[row * 4 + 0]) << "head 0 row " << row;
        EXPECT_FLOAT_EQ(a.data[row * 4 + 1], b.data[row * 4 + 1]) << "head 0 row " << row;
    }
}

TEST(Attention, HeadOutputsKeepTheirColumnOrderWhenConcatenated)
{
    // Same uniform-averaging setup, but split across two heads. Since every
    // head averages its own slice of the same rows, the concatenated result
    // must equal the plain running mean of the input rows -- which also means
    // the heads were stitched back in order rather than shuffled.
    Attention attn = make_uniform_attention("build/test_attention_concat.safetensors", 4, 2);

    Matrix out = attn.forward(Matrix(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                            5.0f, 10.0f, 15.0f, 20.0f}));

    ASSERT_EQ(out.rows, 2);
    ASSERT_EQ(out.cols, 4);
    EXPECT_FLOAT_EQ(out.data[0], 1.0f);
    EXPECT_FLOAT_EQ(out.data[1], 2.0f);
    EXPECT_FLOAT_EQ(out.data[2], 3.0f);
    EXPECT_FLOAT_EQ(out.data[3], 4.0f);
    EXPECT_FLOAT_EQ(out.data[4], 3.0f);
    EXPECT_FLOAT_EQ(out.data[5], 6.0f);
    EXPECT_FLOAT_EQ(out.data[6], 9.0f);
    EXPECT_FLOAT_EQ(out.data[7], 12.0f);
}

TEST(Attention, RejectsHeadCountThatDoesNotDivideDModel)
{
    Attention attn = make_uniform_attention("build/test_attention_baddiv.safetensors", 4, 3);

    EXPECT_THROW(attn.forward(Matrix(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                            5.0f, 6.0f, 7.0f, 8.0f})),
                 std::invalid_argument);
}

TEST(Attention, ThrowsWhenFusedProjectionIsMissing)
{
    std::string path = "build/test_attention_missing.safetensors";
    json header = {
        {"unrelated.weight", {{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {0, 4}}}},
    };
    std::string header_str = header.dump();
    uint64_t header_len = header_str.size();
    float value = 0.0f;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
        out.write(header_str.data(), header_str.size());
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    SafeTensors weights(path);
    EXPECT_THROW(Attention attn(weights, "h.0.", 1), std::runtime_error);
}
