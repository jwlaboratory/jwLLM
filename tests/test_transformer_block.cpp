#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "safetensors.hpp"
#include "transformer_block.hpp"

using json = nlohmann::json;

namespace
{
    struct Tensor
    {
        std::string name;
        std::vector<int> shape;
        std::vector<float> data;
    };

    std::string write_tensors(const std::string &path, const std::vector<Tensor> &tensors)
    {
        json header = json::object();
        uint64_t offset = 0;
        for (const Tensor &t : tensors)
        {
            uint64_t bytes = t.data.size() * sizeof(float);
            header[t.name] = {{"dtype", "F32"}, {"shape", t.shape}, {"data_offsets", {offset, offset + bytes}}};
            offset += bytes;
        }

        std::string header_str = header.dump();
        uint64_t header_len = header_str.size();

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
        out.write(header_str.data(), header_str.size());
        for (const Tensor &t : tensors)
        {
            out.write(reinterpret_cast<const char *>(t.data.data()), t.data.size() * sizeof(float));
        }
        return path;
    }

    std::vector<float> partial_eye(int rows, int cols, float scale = 1.0f)
    {
        std::vector<float> out(rows * cols, 0.0f);
        for (int i = 0; i < rows && i < cols; i++)
        {
            out[i * cols + i] = scale;
        }
        return out;
    }

    // Builds a block where each sublayer's OUTPUT projection is scaled by a
    // caller-chosen factor. Passing 0 silences that branch entirely, so the
    // residual path can be isolated and checked on its own.
    TransformerBlock make_block(const std::string &path,
                                int dmodel,
                                int heads,
                                float attn_out_scale,
                                float mlp_out_scale)
    {
        int hidden = 4 * dmodel;
        SafeTensors weights(write_tensors(path, {
            // layernorms start as the identity transform: gamma 1, beta 0
            {"h.0.ln_1.weight", {dmodel}, std::vector<float>(dmodel, 1.0f)},
            {"h.0.ln_1.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
            {"h.0.ln_2.weight", {dmodel}, std::vector<float>(dmodel, 1.0f)},
            {"h.0.ln_2.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
            // Q and K zeroed so every score ties; V is the identity. Attention
            // then returns the running mean over each token's own past.
            {"h.0.attn.c_attn.weight", {dmodel, 3 * dmodel}, [&] {
                 std::vector<float> w(dmodel * 3 * dmodel, 0.0f);
                 for (int r = 0; r < dmodel; r++)
                 {
                     w[r * 3 * dmodel + 2 * dmodel + r] = 1.0f; // V block only
                 }
                 return w;
             }()},
            {"h.0.attn.c_attn.bias", {3 * dmodel}, std::vector<float>(3 * dmodel, 0.0f)},
            {"h.0.attn.c_proj.weight", {dmodel, dmodel}, partial_eye(dmodel, dmodel, attn_out_scale)},
            {"h.0.attn.c_proj.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
            {"h.0.mlp.c_fc.weight", {dmodel, hidden}, partial_eye(dmodel, hidden)},
            {"h.0.mlp.c_fc.bias", {hidden}, std::vector<float>(hidden, 0.0f)},
            {"h.0.mlp.c_proj.weight", {hidden, dmodel}, partial_eye(hidden, dmodel, mlp_out_scale)},
            {"h.0.mlp.c_proj.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
        }));
        return TransformerBlock(weights, "h.0.", heads);
    }
} // namespace

TEST(TransformerBlock, OutputIsSeqLenByDModel)
{
    TransformerBlock block = make_block("build/test_block_shape.safetensors", 4, 2, 1.0f, 1.0f);

    Matrix out = block.forward(Matrix(3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                             5.0f, 6.0f, 7.0f, 8.0f,
                                             9.0f, 10.0f, 11.0f, 12.0f}));

    EXPECT_EQ(out.rows, 3);
    EXPECT_EQ(out.cols, 4);
}

TEST(TransformerBlock, SilencedSublayersLeaveTheInputUntouched)
{
    // both branches scaled to zero, so forward() must be x + 0 + 0.
    // this is the residual test: it fails loudly if the layernormed value is
    // added back instead of the original x.
    TransformerBlock block = make_block("build/test_block_residual.safetensors", 4, 2, 0.0f, 0.0f);

    Matrix x(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    50.0f, 60.0f, 70.0f, 80.0f});

    Matrix out = block.forward(x);

    for (size_t i = 0; i < x.data.size(); i++)
    {
        EXPECT_NEAR(out.data[i], x.data[i], 1e-4f) << "at index " << i;
    }
}

TEST(TransformerBlock, AddsTheMlpBranchToTheOriginalInput)
{
    // attention silenced, mlp live. the answer is then exactly
    //   x + mlp(layernorm(x))
    // which, with these passthrough mlp weights, is x + gelu(layernorm(x)).
    int dmodel = 4;
    TransformerBlock block = make_block("build/test_block_mlp_only.safetensors", dmodel, 2, 0.0f, 1.0f);

    Matrix x(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    -1.0f, 0.5f, 2.0f, 6.0f});

    Matrix out = block.forward(x);

    Matrix ones(1, dmodel, std::vector<float>(dmodel, 1.0f));
    Matrix zeros(1, dmodel, std::vector<float>(dmodel, 0.0f));
    Matrix expected = x.addition(x.layernorm(ones, zeros).gelu());

    for (size_t i = 0; i < expected.data.size(); i++)
    {
        EXPECT_NEAR(out.data[i], expected.data[i], 1e-4f) << "at index " << i;
    }
}

TEST(TransformerBlock, NormalizesBeforeTheSublayerNotAfter)
{
    // GPT-2 is pre-norm, so a large-magnitude input must survive to the
    // output through the residual. if the block returned the normalized
    // value instead, every row would be squashed to roughly unit scale.
    TransformerBlock block = make_block("build/test_block_prenorm.safetensors", 4, 2, 0.0f, 0.0f);

    Matrix x(1, 4, {100.0f, 200.0f, 300.0f, 400.0f});

    Matrix out = block.forward(x);

    EXPECT_NEAR(out.data[3], 400.0f, 1e-2f);
}

TEST(TransformerBlock, FutureTokensDoNotChangeEarlierRows)
{
    // the causal guarantee, checked through a whole block rather than just
    // the attention op: appending or altering a later token must leave every
    // earlier row bit-identical.
    TransformerBlock block = make_block("build/test_block_causal.safetensors", 4, 2, 1.0f, 1.0f);

    Matrix a(3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f});
    Matrix b(3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    -400.0f, 900.0f, 0.0f, -1.0f});

    Matrix out_a = block.forward(a);
    Matrix out_b = block.forward(b);

    // rows 0 and 1 are the first 8 entries and must match exactly
    for (int i = 0; i < 8; i++)
    {
        EXPECT_NEAR(out_a.data[i], out_b.data[i], 1e-5f) << "at index " << i;
    }
}

TEST(TransformerBlock, ThrowsWhenLayerNormWeightsAreMissing)
{
    int dmodel = 2;
    int hidden = 4 * dmodel;
    SafeTensors weights(write_tensors("build/test_block_missing.safetensors", {
        {"h.0.attn.c_attn.weight", {dmodel, 3 * dmodel}, std::vector<float>(dmodel * 3 * dmodel, 0.0f)},
        {"h.0.attn.c_attn.bias", {3 * dmodel}, std::vector<float>(3 * dmodel, 0.0f)},
        {"h.0.attn.c_proj.weight", {dmodel, dmodel}, partial_eye(dmodel, dmodel)},
        {"h.0.attn.c_proj.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
        {"h.0.mlp.c_fc.weight", {dmodel, hidden}, partial_eye(dmodel, hidden)},
        {"h.0.mlp.c_fc.bias", {hidden}, std::vector<float>(hidden, 0.0f)},
        {"h.0.mlp.c_proj.weight", {hidden, dmodel}, partial_eye(hidden, dmodel)},
        {"h.0.mlp.c_proj.bias", {dmodel}, std::vector<float>(dmodel, 0.0f)},
        // ln_1 and ln_2 deliberately absent
    }));

    EXPECT_THROW(TransformerBlock(weights, "h.0.", 2), std::runtime_error);
}
