#include <gtest/gtest.h>
#include <cmath>

#include "matrix.hpp"

TEST(Matrix, StoresDimensionsAndData)
{
    Matrix matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_EQ(matrix.rows, 2);
    EXPECT_EQ(matrix.cols, 2);
    EXPECT_EQ(matrix.data, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(Matrix, RejectsIncorrectDataSize)
{
    EXPECT_THROW(
        Matrix(2, 2, {1.0f, 2.0f, 3.0f}),
        std::invalid_argument);
}

TEST(Matrix, MultipliesMatrices)
{
    Matrix left(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Matrix right(3, 2, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    Matrix result = left.multiply(right);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 2);
    EXPECT_EQ(result.data, std::vector<float>({58.0f, 64.0f,
                                               139.0f, 154.0f}));
}

TEST(Matrix, RejectsInvalidMultiplicationDimensions)
{
    Matrix left(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Matrix right(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_THROW(left.multiply(right), std::invalid_argument);
}

TEST(Matrix, AddsMatrices)
{
    Matrix left(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});

    Matrix right(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

    Matrix result = left.addition(right);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 2);
    EXPECT_EQ(result.data, std::vector<float>({6.0f, 8.0f,
                                               10.0f, 12.0f}));
}

TEST(Matrix, RejectsInvalidAdditionDimensions)
{
    Matrix left(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix right(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    EXPECT_THROW(left.addition(right), std::invalid_argument);
}

TEST(Matrix, Transposes)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Matrix result = matrix.transpose();

    EXPECT_EQ(result.rows, 3);
    EXPECT_EQ(result.cols, 2);
    EXPECT_EQ(result.data, std::vector<float>({1.0f, 4.0f,
                                               2.0f, 5.0f,
                                               3.0f, 6.0f}));
}

TEST(Matrix, TransposeTwiceReturnsOriginal)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Matrix result = matrix.transpose().transpose();

    EXPECT_EQ(result.rows, matrix.rows);
    EXPECT_EQ(result.cols, matrix.cols);
    EXPECT_EQ(result.data, matrix.data);
}

TEST(Matrix, MultipliesByScalar)
{
    Matrix matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});

    Matrix result = matrix.multiply_scalar(2.5f);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 2);
    EXPECT_EQ(result.data, std::vector<float>({2.5f, 5.0f, 7.5f, 10.0f}));
}

TEST(Matrix, MultipliesByNegativeScalar)
{
    Matrix matrix(1, 3, {1.0f, -2.0f, 3.0f});

    Matrix result = matrix.multiply_scalar(-1.0f);

    EXPECT_EQ(result.data, std::vector<float>({-1.0f, 2.0f, -3.0f}));
}

TEST(Matrix, GeluOfZeroIsZero)
{
    Matrix matrix(1, 1, {0.0f});

    Matrix result = matrix.gelu();

    EXPECT_NEAR(result.data[0], 0.0f, 1e-4f);
}

TEST(Matrix, GeluMatchesKnownValues)
{
    Matrix matrix(1, 2, {1.0f, -1.0f});

    Matrix result = matrix.gelu();

    // GPT-2 uses the tanh approximation of GELU.
    EXPECT_NEAR(result.data[0], 0.8412f, 1e-3f);
    EXPECT_NEAR(result.data[1], -0.1588f, 1e-3f);
}

TEST(Matrix, GeluOfLargeNegativeApproachesZero)
{
    Matrix matrix(1, 1, {-10.0f});

    Matrix result = matrix.gelu();

    EXPECT_NEAR(result.data[0], 0.0f, 1e-3f);
}

TEST(Matrix, BroadcastAddRowAddsToEveryRow)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix row(1, 3, {10.0f, 20.0f, 30.0f});

    Matrix result = matrix.broadcast_add_row(row);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 3);
    EXPECT_EQ(result.data, std::vector<float>({11.0f, 22.0f, 33.0f,
                                               14.0f, 25.0f, 36.0f}));
}

TEST(Matrix, BroadcastAddRowRejectsWrongShape)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix notARow(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix wrongCols(1, 2, {1.0f, 2.0f});

    EXPECT_THROW(matrix.broadcast_add_row(notARow), std::invalid_argument);
    EXPECT_THROW(matrix.broadcast_add_row(wrongCols), std::invalid_argument);
}

TEST(Matrix, BroadcastMultiplyRowMultipliesEveryRow)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix row(1, 3, {2.0f, 0.5f, 1.0f});

    Matrix result = matrix.broadcast_multiply_row(row);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 3);
    EXPECT_EQ(result.data, std::vector<float>({2.0f, 1.0f, 3.0f,
                                               8.0f, 2.5f, 6.0f}));
}

TEST(Matrix, BroadcastMultiplyRowRejectsWrongShape)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Matrix wrongCols(1, 2, {1.0f, 2.0f});

    EXPECT_THROW(matrix.broadcast_multiply_row(wrongCols), std::invalid_argument);
}

TEST(Matrix, SoftmaxRowsSumsToOne)
{
    Matrix matrix(2, 3, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});

    Matrix result = matrix.softmax_rows();

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 3);

    float row0_sum = result.data[0] + result.data[1] + result.data[2];
    float row1_sum = result.data[3] + result.data[4] + result.data[5];
    EXPECT_NEAR(row0_sum, 1.0f, 1e-4f);
    EXPECT_NEAR(row1_sum, 1.0f, 1e-4f);

    // uniform input -> uniform distribution
    EXPECT_NEAR(result.data[3], 1.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.data[4], 1.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(result.data[5], 1.0f / 3.0f, 1e-4f);

    // largest logit in a row gets the largest probability
    EXPECT_GT(result.data[2], result.data[1]);
    EXPECT_GT(result.data[1], result.data[0]);
}

TEST(Matrix, SoftmaxRowsIsStableForLargeValues)
{
    Matrix matrix(1, 3, {1000.0f, 1001.0f, 1002.0f});

    Matrix result = matrix.softmax_rows();

    for (float value : result.data)
    {
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
    }
    float row_sum = result.data[0] + result.data[1] + result.data[2];
    EXPECT_NEAR(row_sum, 1.0f, 1e-3f);
}

TEST(Matrix, SliceColsExtractsSubRange)
{
    Matrix matrix(2, 4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    Matrix result = matrix.slice_cols(1, 2);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 2);
    EXPECT_EQ(result.data, std::vector<float>({2.0f, 3.0f,
                                               6.0f, 7.0f}));
}

TEST(Matrix, SliceColsRejectsOutOfRange)
{
    Matrix matrix(2, 4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    EXPECT_THROW(matrix.slice_cols(-1, 2), std::invalid_argument);
    EXPECT_THROW(matrix.slice_cols(3, 2), std::invalid_argument);
    EXPECT_THROW(matrix.slice_cols(0, 0), std::invalid_argument);
}

TEST(Matrix, ConcatColsJoinsMatrices)
{
    Matrix left(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix right(2, 1, {5.0f, 6.0f});

    Matrix result = left.concat_cols(right);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 3);
    EXPECT_EQ(result.data, std::vector<float>({1.0f, 2.0f, 5.0f,
                                               3.0f, 4.0f, 6.0f}));
}

TEST(Matrix, ConcatColsRejectsMismatchedRows)
{
    Matrix left(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix right(3, 1, {5.0f, 6.0f, 7.0f});

    EXPECT_THROW(left.concat_cols(right), std::invalid_argument);
}

TEST(Matrix, SliceColsThenConcatColsReturnsOriginal)
{
    Matrix matrix(2, 4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    Matrix left = matrix.slice_cols(0, 2);
    Matrix right = matrix.slice_cols(2, 2);
    Matrix rejoined = left.concat_cols(right);

    EXPECT_EQ(rejoined.rows, matrix.rows);
    EXPECT_EQ(rejoined.cols, matrix.cols);
    EXPECT_EQ(rejoined.data, matrix.data);
}
TEST(Matrix, LayernormNormalizesEachRowToZeroMeanUnitVariance)
{
    Matrix x(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    10.0f, 10.0f, 10.0f, 10.0f});
    Matrix gamma(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
    Matrix beta(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});

    Matrix result = x.layernorm(gamma, beta);

    EXPECT_EQ(result.rows, 2);
    EXPECT_EQ(result.cols, 4);

    // row 0: mean 2.5, var 1.25 (population, not sample), std ~1.1180
    float s = std::sqrt(1.25f + 1e-5f);
    EXPECT_NEAR(result.data[0], -1.5f / s, 1e-4f);
    EXPECT_NEAR(result.data[1], -0.5f / s, 1e-4f);
    EXPECT_NEAR(result.data[2], 0.5f / s, 1e-4f);
    EXPECT_NEAR(result.data[3], 1.5f / s, 1e-4f);

    // row 1 is constant: var 0, so eps keeps us from dividing by zero and
    // every entry comes out 0
    for (int g = 0; g < 4; g++)
    {
        EXPECT_NEAR(result.data[4 + g], 0.0f, 1e-6f);
    }
}

TEST(Matrix, LayernormRowsAreIndependent)
{
    // same row 0 as above, but a different row 1 must not change row 0's result
    Matrix a(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    10.0f, 10.0f, 10.0f, 10.0f});
    Matrix b(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    -100.0f, 0.0f, 50.0f, 7.0f});
    Matrix gamma(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
    Matrix beta(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});

    Matrix ra = a.layernorm(gamma, beta);
    Matrix rb = b.layernorm(gamma, beta);

    for (int g = 0; g < 4; g++)
    {
        EXPECT_NEAR(ra.data[g], rb.data[g], 1e-6f);
    }
}

TEST(Matrix, LayernormAppliesGammaAndBetaPerColumn)
{
    Matrix x(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix gamma(1, 4, {2.0f, 0.0f, -1.0f, 1.0f});
    Matrix beta(1, 4, {0.0f, 5.0f, 1.0f, -1.0f});

    Matrix plain = x.layernorm(Matrix(1, 4, {1.0f, 1.0f, 1.0f, 1.0f}),
                               Matrix(1, 4, {0.0f, 0.0f, 0.0f, 0.0f}));
    Matrix result = x.layernorm(gamma, beta);

    for (int g = 0; g < 4; g++)
    {
        EXPECT_NEAR(result.data[g], plain.data[g] * gamma.data[g] + beta.data[g], 1e-5f);
    }
}

TEST(Matrix, LayernormDoesNotModifyInput)
{
    Matrix x(1, 3, {3.0f, 6.0f, 9.0f});
    Matrix gamma(1, 3, {1.0f, 1.0f, 1.0f});
    Matrix beta(1, 3, {0.0f, 0.0f, 0.0f});

    x.layernorm(gamma, beta);

    EXPECT_EQ(x.data, std::vector<float>({3.0f, 6.0f, 9.0f}));
}

TEST(Matrix, LayernormRejectsMismatchedGammaBetaWidth)
{
    Matrix x(2, 4, {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f});
    Matrix gamma3(1, 3, {1.0f, 1.0f, 1.0f});
    Matrix beta4(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});

    EXPECT_THROW(x.layernorm(gamma3, beta4), std::invalid_argument);
}

TEST(Matrix, LayernormMatchesGPT2ScaleBehaviour)
{
    // layernorm is scale invariant: scaling every entry in a row by a
    // constant should give the same normalized output (up to eps)
    Matrix x(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    Matrix xs = x.multiply_scalar(1000.0f);
    Matrix gamma(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
    Matrix beta(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});

    Matrix r = x.layernorm(gamma, beta);
    Matrix rs = xs.layernorm(gamma, beta);

    for (int g = 0; g < 4; g++)
    {
        EXPECT_NEAR(r.data[g], rs.data[g], 1e-3f);
    }
}

TEST(Matrix, MaskCausalBlanksOutTheUpperTriangle)
{
    Matrix scores(3, 3, {1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f,
                         7.0f, 8.0f, 9.0f});

    Matrix result = scores.mask_causal();

    EXPECT_EQ(result.rows, 3);
    EXPECT_EQ(result.cols, 3);

    // col > row is the future, and must become very negative. the test does
    // not pin the exact constant, only that softmax will flush it to zero.
    EXPECT_LT(result.data[1], -1e8f); // row 0, col 1
    EXPECT_LT(result.data[2], -1e8f); // row 0, col 2
    EXPECT_LT(result.data[5], -1e8f); // row 1, col 2
}

TEST(Matrix, MaskCausalLeavesThePastAndDiagonalUntouched)
{
    Matrix scores(3, 3, {1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f,
                         7.0f, 8.0f, 9.0f});

    Matrix result = scores.mask_causal();

    // diagonal: every position can always see itself
    EXPECT_FLOAT_EQ(result.data[0], 1.0f); // row 0, col 0
    EXPECT_FLOAT_EQ(result.data[4], 5.0f); // row 1, col 1
    EXPECT_FLOAT_EQ(result.data[8], 9.0f); // row 2, col 2

    // strictly below the diagonal: the past, passed through unchanged
    EXPECT_FLOAT_EQ(result.data[3], 4.0f); // row 1, col 0
    EXPECT_FLOAT_EQ(result.data[6], 7.0f); // row 2, col 0
    EXPECT_FLOAT_EQ(result.data[7], 8.0f); // row 2, col 1
}

TEST(Matrix, MaskCausalRejectsNonSquare)
{
    // a score matrix is always seq x seq; anything else is a bug upstream
    Matrix wrong(2, 3, {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f});

    EXPECT_THROW(wrong.mask_causal(), std::invalid_argument);
}

TEST(Matrix, MaskCausalDoesNotModifyInput)
{
    Matrix scores(2, 2, {1.0f, 2.0f,
                         3.0f, 4.0f});

    scores.mask_causal();

    EXPECT_EQ(scores.data, std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(Matrix, MaskCausalIsANoOpForASingleToken)
{
    Matrix scores(1, 1, {7.0f});

    Matrix result = scores.mask_causal();

    EXPECT_FLOAT_EQ(result.data[0], 7.0f);
}

TEST(Matrix, MaskCausalThenSoftmaxGivesLowerTriangularWeights)
{
    // the real contract: mask -> softmax_rows should leave every row a
    // probability distribution over that row's own past only.
    Matrix scores(3, 3, {1.0f, 9.0f, 9.0f,
                         1.0f, 1.0f, 9.0f,
                         1.0f, 1.0f, 1.0f});

    Matrix weights = scores.mask_causal().softmax_rows();

    // future positions get no weight at all, even though their raw scores
    // were the largest in the row
    EXPECT_NEAR(weights.data[1], 0.0f, 1e-6f);
    EXPECT_NEAR(weights.data[2], 0.0f, 1e-6f);
    EXPECT_NEAR(weights.data[5], 0.0f, 1e-6f);

    // token 0 has only itself to look at, so it must put all its weight there
    EXPECT_NEAR(weights.data[0], 1.0f, 1e-6f);

    // token 1 splits evenly between positions 0 and 1 (equal scores)
    EXPECT_NEAR(weights.data[3], 0.5f, 1e-6f);
    EXPECT_NEAR(weights.data[4], 0.5f, 1e-6f);

    // token 2 splits evenly across all three
    for (int g = 0; g < 3; g++)
    {
        EXPECT_NEAR(weights.data[6 + g], 1.0f / 3.0f, 1e-6f);
    }

    // and every row is still a distribution
    for (int i = 0; i < 3; i++)
    {
        float sum = 0;
        for (int g = 0; g < 3; g++)
        {
            sum += weights.data[i * 3 + g];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST(Matrix, MaskCausalMakesEarlierRowsIgnoreLaterTokens)
{
    // changing a score that only a *later* row can see must not change any
    // earlier row's weights. this is the invariant that makes KV caching work.
    Matrix a(3, 3, {1.0f, 5.0f, 5.0f,
                    2.0f, 3.0f, 5.0f,
                    1.0f, 2.0f, 3.0f});
    Matrix b(3, 3, {1.0f, -50.0f, 99.0f,
                    2.0f, 3.0f, -7.0f,
                    1.0f, 2.0f, 3.0f});

    Matrix wa = a.mask_causal().softmax_rows();
    Matrix wb = b.mask_causal().softmax_rows();

    // rows 0 and 1 differ only in their masked-out columns, so their
    // resulting weights must be identical
    for (int g = 0; g < 6; g++)
    {
        EXPECT_NEAR(wa.data[g], wb.data[g], 1e-6f);
    }
}
