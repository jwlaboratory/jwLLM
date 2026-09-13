#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "safetensors.hpp"

using json = nlohmann::json;

namespace
{
    // A 2-D tensor and a 1-D tensor, laid out back to back.
    const std::vector<float> kWeight = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};        // shape {2, 3}
    const std::vector<float> kBias = {7.0f, 8.0f, 9.0f}; // shape {3}

    std::string write_fixture()
    {
        std::string path = "build/test_safetensors_fixture.safetensors";

        uint64_t weight_bytes = kWeight.size() * sizeof(float);
        uint64_t bias_bytes = kBias.size() * sizeof(float);

        json header = {
            {"__metadata__", {{"format", "pt"}}},
            {"layer.weight", {{"dtype", "F32"}, {"shape", {2, 3}}, {"data_offsets", {0, weight_bytes}}}},
            {"layer.bias", {{"dtype", "F32"}, {"shape", {3}}, {"data_offsets", {weight_bytes, weight_bytes + bias_bytes}}}},
        };
        std::string header_str = header.dump();
        uint64_t header_len = header_str.size();

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
        out.write(header_str.data(), header_str.size());
        out.write(reinterpret_cast<const char *>(kWeight.data()), weight_bytes);
        out.write(reinterpret_cast<const char *>(kBias.data()), bias_bytes);

        return path;
    }
} // namespace

TEST(SafeTensors, ThrowsOnMissingFile)
{
    EXPECT_THROW(SafeTensors("data/does_not_exist.safetensors"), std::runtime_error);
}

TEST(SafeTensors, HasReportsTensorNames)
{
    SafeTensors st(write_fixture());

    EXPECT_TRUE(st.has("layer.weight"));
    EXPECT_TRUE(st.has("layer.bias"));
    EXPECT_FALSE(st.has("layer.nope"));
}

TEST(SafeTensors, GetReads2DTensor)
{
    SafeTensors st(write_fixture());

    Matrix w = st.get("layer.weight");

    EXPECT_EQ(w.rows, 2);
    EXPECT_EQ(w.cols, 3);
    EXPECT_EQ(w.data, kWeight);
}

TEST(SafeTensors, GetReads1DTensorAsSingleRow)
{
    SafeTensors st(write_fixture());

    Matrix b = st.get("layer.bias");

    EXPECT_EQ(b.rows, 1);
    EXPECT_EQ(b.cols, 3);
    EXPECT_EQ(b.data, kBias);
}

TEST(SafeTensors, GetCanBeCalledInAnyOrderAndRepeatedly)
{
    SafeTensors st(write_fixture());

    // Second tensor first, then first, then second again: seeking must be
    // absolute, not relative to the previous read.
    Matrix b1 = st.get("layer.bias");
    Matrix w = st.get("layer.weight");
    Matrix b2 = st.get("layer.bias");

    EXPECT_EQ(w.data, kWeight);
    EXPECT_EQ(b1.data, kBias);
    EXPECT_EQ(b2.data, kBias);
}

TEST(SafeTensors, GetThrowsOnMissingName)
{
    SafeTensors st(write_fixture());

    EXPECT_THROW(st.get("layer.nope"), std::runtime_error);
}

TEST(SafeTensors, ReadsRealGPT2Weights)
{
    SafeTensors st("data/model.safetensors");

    Matrix wte = st.get("wte.weight");
    EXPECT_EQ(wte.rows, 50257);
    EXPECT_EQ(wte.cols, 768);

    // a per-layer bias comes back as a single row
    Matrix ln_b = st.get("h.0.ln_1.bias");
    EXPECT_EQ(ln_b.rows, 1);
    EXPECT_EQ(ln_b.cols, 768);
}
