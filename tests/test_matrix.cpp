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