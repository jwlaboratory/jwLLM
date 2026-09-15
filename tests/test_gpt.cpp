#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "gpt.hpp"
#include "matrix.hpp"
#include "safetensors.hpp"

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

    // Tiny GPT: vocab 5, d_model 4, 2 heads, 8 positions, n_layers layers.
    // wte rows are one-hot-ish so the tied output head maps token i's
    // embedding back to a peak at logit i.
    const int kVocab = 5;
    const int kDModel = 4;
    const int kHeads = 2;
    const int kPositions = 8;

    std::vector<Tensor> layer_tensors(int layer)
    {
        std::string p = "h." + std::to_string(layer) + ".";
        int hidden = 4 * kDModel;
        return {
            {p + "ln_1.weight", {kDModel}, std::vector<float>(kDModel, 1.0f)},
            {p + "ln_1.bias", {kDModel}, std::vector<float>(kDModel, 0.0f)},
            {p + "ln_2.weight", {kDModel}, std::vector<float>(kDModel, 1.0f)},
            {p + "ln_2.bias", {kDModel}, std::vector<float>(kDModel, 0.0f)},
            {p + "attn.c_attn.weight", {kDModel, 3 * kDModel}, std::vector<float>(kDModel * 3 * kDModel, 0.0f)},
            {p + "attn.c_attn.bias", {3 * kDModel}, std::vector<float>(3 * kDModel, 0.0f)},
            {p + "attn.c_proj.weight", {kDModel, kDModel}, std::vector<float>(kDModel * kDModel, 0.0f)},
            {p + "attn.c_proj.bias", {kDModel}, std::vector<float>(kDModel, 0.0f)},
            {p + "mlp.c_fc.weight", {kDModel, hidden}, std::vector<float>(kDModel * hidden, 0.0f)},
            {p + "mlp.c_fc.bias", {hidden}, std::vector<float>(hidden, 0.0f)},
            {p + "mlp.c_proj.weight", {hidden, kDModel}, std::vector<float>(hidden * kDModel, 0.0f)},
            {p + "mlp.c_proj.bias", {kDModel}, std::vector<float>(kDModel, 0.0f)},
        };
    }

    std::string write_model(const std::string &path, int n_layers)
    {
        std::vector<Tensor> tensors = {
            // wte: [vocab, d_model]. rows 0..3 are scaled unit vectors, row 4 is all-ones.
            {"wte.weight", {kVocab, kDModel}, {
                                                  2.0f, 0.0f, 0.0f, 0.0f,
                                                  0.0f, 2.0f, 0.0f, 0.0f,
                                                  0.0f, 0.0f, 2.0f, 0.0f,
                                                  0.0f, 0.0f, 0.0f, 2.0f,
                                                  1.0f, 1.0f, 1.0f, 1.0f,
                                              }},
            {"wpe.weight", {kPositions, kDModel}, std::vector<float>(kPositions * kDModel, 0.0f)},
            {"ln_f.weight", {kDModel}, std::vector<float>(kDModel, 1.0f)},
            {"ln_f.bias", {kDModel}, std::vector<float>(kDModel, 0.0f)},
        };
        for (int i = 0; i < n_layers; i++)
        {
            std::vector<Tensor> layer = layer_tensors(i);
            tensors.insert(tensors.end(), layer.begin(), layer.end());
        }
        return write_tensors(path, tensors);
    }
} // namespace

TEST(GPT, LogitsAreSeqLenByVocab)
{
    SafeTensors weights(write_model("build/test_gpt_shape.safetensors", 2));
    GPT model(weights, 2, kHeads);

    Matrix logits = model.forward({0, 1, 2});
    EXPECT_EQ(logits.rows, 3);
    EXPECT_EQ(logits.cols, kVocab);
    EXPECT_EQ(model.max_context(), kPositions);
}

TEST(GPT, ZeroedBlocksPassResidualThroughToTiedHead)
{
    // Every block's output projections are zero, so each block is the identity
    // on the residual stream. With zero wpe and identity ln_f, the final
    // hidden state is layernorm(wte[id]); its largest dot product with wte
    // rows is the row for id itself (a scaled unit vector along the same axis).
    SafeTensors weights(write_model("build/test_gpt_identity.safetensors", 1));
    GPT model(weights, 1, kHeads);

    for (int id = 0; id < 4; id++)
    {
        EXPECT_EQ(model.next_token({id}), id) << "token " << id;
    }
}

TEST(GPT, GenerateAppendsExactlyMaxNewTokens)
{
    SafeTensors weights(write_model("build/test_gpt_generate.safetensors", 1));
    GPT model(weights, 1, kHeads);

    std::vector<int> out = model.generate({3, 1}, 4);
    ASSERT_EQ(out.size(), 6u);
    EXPECT_EQ(out[0], 3);
    EXPECT_EQ(out[1], 1);
    for (int id : out)
    {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, kVocab);
    }
}

TEST(GPT, GenerateStopsAtContextLimit)
{
    SafeTensors weights(write_model("build/test_gpt_context.safetensors", 1));
    GPT model(weights, 1, kHeads);

    std::vector<int> prompt(kPositions - 2, 0);
    std::vector<int> out = model.generate(prompt, 100);
    EXPECT_EQ(out.size(), static_cast<size_t>(kPositions));
}

TEST(GPT, ForwardRejectsEmptyInput)
{
    SafeTensors weights(write_model("build/test_gpt_empty.safetensors", 1));
    GPT model(weights, 1, kHeads);
    EXPECT_THROW(model.forward({}), std::invalid_argument);
}

TEST(GPT, ThrowsWhenLayerWeightsAreMissing)
{
    // file only has h.0, asking for 2 layers must fail loudly
    SafeTensors weights(write_model("build/test_gpt_missing_layer.safetensors", 1));
    EXPECT_ANY_THROW(GPT(weights, 2, kHeads));
}
