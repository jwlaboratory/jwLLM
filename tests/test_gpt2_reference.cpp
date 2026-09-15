#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gpt.hpp"
#include "matrix.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"

// Golden-value tests against official GPT-2 small.
//
// Token ids come from tiktoken's "gpt2" encoding. Logits and greedy
// continuations come from an independent numpy implementation of the
// HuggingFace GPT-2 forward pass reading the same data/model.safetensors
// (Conv1D layout, tanh gelu, layernorm eps 1e-5). The C++ model was measured
// at max |diff| ~4e-4 on logits of magnitude ~170, so kLogitTol is loose
// enough for float summation-order noise but tight enough to catch a real
// bug in any layer.
//
// These run the full 12-layer model on real weights, so they take a while.

namespace
{
    const float kLogitTol = 0.01f;

    struct Model
    {
        Tokenizer tok{"data/vocab.json", "data/merges.txt"};
        SafeTensors weights{"data/model.safetensors"};
        GPT gpt{weights};
    };

    // load the ~500MB of weights once for the whole file
    Model &model()
    {
        static std::unique_ptr<Model> m = std::make_unique<Model>();
        return *m;
    }

    const std::vector<int> kFranceIds = {464, 3139, 286, 4881, 318};
    const std::vector<int> kTuringIds = {36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716};

    std::vector<int> top_k(const Matrix &logits, int row, int k)
    {
        std::vector<int> ids(logits.cols);
        for (int j = 0; j < logits.cols; j++)
            ids[j] = j;
        const float *r = logits.data.data() + row * logits.cols;
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                          [r](int a, int b)
                          { return r[a] > r[b]; });
        ids.resize(k);
        return ids;
    }
} // namespace

TEST(GPT2Reference, TokenizerMatchesTiktoken)
{
    Tokenizer &tok = model().tok;
    EXPECT_EQ(tok.encode("The capital of France is"), kFranceIds);
    EXPECT_EQ(tok.encode("Hello world!!!"), (std::vector<int>{15496, 995, 10185}));
    EXPECT_EQ(tok.encode("Alan Turing theorized that computers would one day become"), kTuringIds);

    EXPECT_EQ(tok.decode(kFranceIds), "The capital of France is");
    EXPECT_EQ(tok.decode({262, 3139, 286, 262, 4141}), " the capital of the French");
}

TEST(GPT2Reference, LogitsMatchReferenceForwardPass)
{
    Matrix logits = model().gpt.forward(kFranceIds);

    ASSERT_EQ(logits.rows, 5);
    ASSERT_EQ(logits.cols, 50257);

    // argmax of every row, not just the last, so a bug in the causal mask shows up
    std::vector<int> expected_argmax = {198, 286, 262, 11, 262};
    for (int r = 0; r < logits.rows; r++)
    {
        EXPECT_EQ(top_k(logits, r, 1)[0], expected_argmax[r]) << "row " << r;
    }

    // top-5 ordering on first and last rows
    EXPECT_EQ(top_k(logits, 0, 5), (std::vector<int>{198, 262, 366, 11, 257}));
    EXPECT_EQ(top_k(logits, 4, 5), (std::vector<int>{262, 783, 257, 4881, 6342}));

    // raw logit values, row 0 (only sees the first token)
    const float *row0 = logits.data.data();
    EXPECT_NEAR(row0[198], -31.864f, kLogitTol);
    EXPECT_NEAR(row0[262], -32.174f, kLogitTol);
    EXPECT_NEAR(row0[366], -32.3702f, kLogitTol);

    // raw logit values, row 4 (sees the whole prompt)
    const float *row4 = logits.data.data() + 4 * logits.cols;
    EXPECT_NEAR(row4[262], -100.2498f, kLogitTol);
    EXPECT_NEAR(row4[783], -100.8176f, kLogitTol);
    EXPECT_NEAR(row4[257], -100.8555f, kLogitTol);
    EXPECT_NEAR(row4[4881], -101.2102f, kLogitTol);
    EXPECT_NEAR(row4[6342], -101.2143f, kLogitTol);
    // and a few far from the top, so the whole distribution is checked not just the peak
    EXPECT_NEAR(row4[0], -108.9543f, kLogitTol);
    EXPECT_NEAR(row4[1000], -113.6523f, kLogitTol);
    EXPECT_NEAR(row4[50256], -110.3779f, kLogitTol);
}

TEST(GPT2Reference, LogitsMatchReferenceOnLongerPrompt)
{
    Matrix logits = model().gpt.forward(kTuringIds);

    ASSERT_EQ(logits.rows, 10);
    ASSERT_EQ(logits.cols, 50257);

    std::vector<int> expected_argmax = {11, 11, 1143, 326, 262, 714, 307, 1110, 307, 262};
    for (int r = 0; r < logits.rows; r++)
    {
        EXPECT_EQ(top_k(logits, r, 1)[0], expected_argmax[r]) << "row " << r;
    }

    EXPECT_EQ(top_k(logits, 9, 5), (std::vector<int>{262, 257, 366, 517, 2208}));

    const float *row9 = logits.data.data() + 9 * logits.cols;
    EXPECT_NEAR(row9[262], -102.8067f, kLogitTol);
    EXPECT_NEAR(row9[257], -103.7922f, kLogitTol);
    EXPECT_NEAR(row9[366], -104.1135f, kLogitTol);
    EXPECT_NEAR(row9[517], -104.598f, kLogitTol);
    EXPECT_NEAR(row9[2208], -104.855f, kLogitTol);
}

TEST(GPT2Reference, GreedyGenerationMatchesReference)
{
    Model &m = model();

    std::vector<int> france = m.gpt.generate(kFranceIds, 5);
    ASSERT_EQ(france.size(), kFranceIds.size() + 5);
    std::vector<int> france_new(france.begin() + kFranceIds.size(), france.end());
    EXPECT_EQ(france_new, (std::vector<int>{262, 3139, 286, 262, 4141}));
    EXPECT_EQ(m.tok.decode(france_new), " the capital of the French");

    std::vector<int> turing = m.gpt.generate(kTuringIds, 5);
    ASSERT_EQ(turing.size(), kTuringIds.size() + 5);
    std::vector<int> turing_new(turing.begin() + kTuringIds.size(), turing.end());
    EXPECT_EQ(turing_new, (std::vector<int>{262, 749, 3665, 8217, 319}));
    EXPECT_EQ(m.tok.decode(turing_new), " the most powerful machines on");
}
