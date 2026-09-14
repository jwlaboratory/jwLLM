#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "mlp.hpp"
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

    // Writes a safetensors file holding exactly the given tensors, laid out
    // back to back in order.
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

    // A [rows, cols] matrix with ones on the main diagonal. Not square in
    // general: used to widen dmodel -> 4*dmodel by padding with zeros, and to
    // narrow back again.
    std::vector<float> partial_eye(int rows, int cols, float scale = 1.0f)
    {
        std::vector<float> out(rows * cols, 0.0f);
        for (int i = 0; i < rows && i < cols; i++)
        {
            out[i * cols + i] = scale;
        }
        return out;
    }

    // c_fc copies x into the first dmodel of the 4*dmodel columns and zeroes
    // the rest; c_proj copies the first dmodel back. gelu(0) is 0, so the
    // padding contributes nothing and forward() reduces to exactly gelu(x).
    MLP make_passthrough_mlp(const std::string &path,
                             int dmodel,
                             const std::vector<float> &fc_bias = {},
                             const std::vector<float> &proj_bias = {})
    {
        int hidden = 4 * dmodel;
        SafeTensors weights(write_tensors(path, {
            {"h.0.mlp.c_fc.weight", {dmodel, hidden}, partial_eye(dmodel, hidden)},
            {"h.0.mlp.c_fc.bias", {hidden}, fc_bias.empty() ? std::vector<float>(hidden, 0.0f) : fc_bias},
            {"h.0.mlp.c_proj.weight", {hidden, dmodel}, partial_eye(hidden, dmodel)},
            {"h.0.mlp.c_proj.bias", {dmodel}, proj_bias.empty() ? std::vector<float>(dmodel, 0.0f) : proj_bias},
        }));
        return MLP(weights, "h.0.");
    }
} // namespace

TEST(MLP, OutputIsSeqLenByDModel)
{
    MLP mlp = make_passthrough_mlp("build/test_mlp_shape.safetensors", 4);

    Matrix out = mlp.forward(Matrix(3, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f,
                                           9.0f, 10.0f, 11.0f, 12.0f}));

    // widened to 16 internally, but the caller only ever sees dmodel back
    EXPECT_EQ(out.rows, 3);
    EXPECT_EQ(out.cols, 4);
}

TEST(MLP, PassthroughWeightsReduceToGelu)
{
    MLP mlp = make_passthrough_mlp("build/test_mlp_gelu.safetensors", 4);

    Matrix x(2, 4, {1.0f, -2.0f, 0.5f, 0.0f,
                    -0.5f, 3.0f, -1.0f, 2.0f});

    Matrix out = mlp.forward(x);
    Matrix expected = x.gelu();

    ASSERT_EQ(out.data.size(), expected.data.size());
    for (size_t i = 0; i < expected.data.size(); i++)
    {
        EXPECT_NEAR(out.data[i], expected.data[i], 1e-5f) << "at index " << i;
    }
}

TEST(MLP, ActivationIsNonLinear)
{
    // guards against forgetting gelu entirely: with these weights a purely
    // linear path would hand back x unchanged, and gelu must suppress the
    // negatives while leaving large positives nearly alone.
    MLP mlp = make_passthrough_mlp("build/test_mlp_nonlinear.safetensors", 2);

    Matrix out = mlp.forward(Matrix(1, 2, {-3.0f, 4.0f}));

    EXPECT_GT(out.data[0], -3.0f); // negative pulled up toward zero
    EXPECT_LT(out.data[0], 0.0f);  // but still negative
    EXPECT_NEAR(out.data[1], 4.0f, 0.01f); // large positive passes through
}

TEST(MLP, BiasesAreBroadcastAcrossTokens)
{
    int dmodel = 2;
    // c_proj bias shifts every output row by the same amount
    MLP mlp = make_passthrough_mlp("build/test_mlp_bias.safetensors", dmodel,
                                   std::vector<float>(4 * dmodel, 0.0f),
                                   {10.0f, 20.0f});

    Matrix x(3, 2, {0.0f, 0.0f,
                    0.0f, 0.0f,
                    0.0f, 0.0f});

    Matrix out = mlp.forward(x);

    // gelu(0) is 0, so every row is just the projection bias
    for (int i = 0; i < 3; i++)
    {
        EXPECT_NEAR(out.data[i * 2 + 0], 10.0f, 1e-5f);
        EXPECT_NEAR(out.data[i * 2 + 1], 20.0f, 1e-5f);
    }
}

TEST(MLP, TokensAreIndependent)
{
    // the defining difference from attention: the MLP never moves information
    // between positions, so changing one token cannot affect another's row.
    MLP mlp = make_passthrough_mlp("build/test_mlp_independent.safetensors", 2);

    Matrix a(2, 2, {1.0f, 2.0f,
                    3.0f, 4.0f});
    Matrix b(2, 2, {1.0f, 2.0f,
                    -99.0f, 77.0f});

    Matrix out_a = mlp.forward(a);
    Matrix out_b = mlp.forward(b);

    // row 0 is identical in both inputs, so it must be identical in both
    // outputs no matter what row 1 does
    EXPECT_NEAR(out_a.data[0], out_b.data[0], 1e-6f);
    EXPECT_NEAR(out_a.data[1], out_b.data[1], 1e-6f);
}

TEST(MLP, ThrowsWhenWeightsAreMissing)
{
    SafeTensors weights(write_tensors("build/test_mlp_missing.safetensors", {
        {"h.0.mlp.c_fc.weight", {2, 8}, std::vector<float>(16, 0.0f)},
    }));

    EXPECT_THROW(MLP(weights, "h.0."), std::runtime_error);
}
