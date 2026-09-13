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