#include <gtest/gtest.h>

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