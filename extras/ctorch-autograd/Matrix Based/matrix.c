#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

struct Matrix *createMatrix(int rows, int cols, float *data, bool isUpdatable, struct Matrix *child1, struct Matrix *child2, char op)
{
    struct Matrix *out = malloc(sizeof(struct Matrix));

    out->rows = rows;
    out->cols = cols;

    out->data = data;
    out->grads = calloc(rows * cols, sizeof(float));

    out->isVisited = false;
    out->isUpdatable = isUpdatable;

    out->child1 = child1;
    out->child2 = child2;
    out->op = op;

    return out;
}
struct Matrix *createRandMatrix(int rows, int cols, bool isUpdatable)
{
    float *data = malloc(sizeof(float) * rows * cols);

    for (int i = 0; i < rows * cols; i++)
    {
        float randVal = ((float)(rand() % 100) + 1) / 100.0f;
        data[i] = randVal;
    }

    return createMatrix(rows, cols, data, isUpdatable, NULL, NULL, 'l');
}

float getMatrixValue(struct Matrix *mat, int row, int col)
{
    int indx = mat->cols * row + col;
    return mat->data[indx];
}

void setMatrixValue(struct Matrix *mat, int row, int col, float data)
{
    int indx = mat->cols * row + col;
    mat->data[indx] = data;
}

struct Matrix *multiplyMatrix(struct Matrix *A, struct Matrix *B)
{

    if (A->cols != B->rows)
    {
        printf("[matmul] incompatible size error");
        return NULL;
    }

    int rows = A->rows;
    int cols = B->cols;
    float *data = calloc(rows * cols, sizeof(float));
    struct Matrix *res = createMatrix(rows, cols, data, false, A, B, '@');

    // shd we assume row major or col major? assume row major
    // note that actual index will begin from 0

    // [ a11 a12
    //  a21,a22 ]

    // opt a[a11 a12 a21 a22]
    // opt b[a11 a21 a12 a22]

    // a=3x2
    // b=2x3

    //[a11Xb11 + a12XB12, a11XB12 + a12xb22, a11Xb13+a12xb23]

    for (int i = 0; i < res->rows; i++)
    {
        for (int g = 0; g < res->cols; g++)
        {
            // i = row, g = col in new matrix
            // formula for a a=3x2, b=2x3 ;
            // generalized, we loop for the pattern

            // float sum = 0;

            for (int q = 0; q < A->cols; q++)
            {
                // sum += getMatrixValue(A, i, q) * getMatrixValue(B, q, g);
                data[i * res->cols + g] += getMatrixValue(A, i, q) * getMatrixValue(B, q, g); // naive way without accumulator
                // wed keep going as so
                // + getMatrixValue(A, i, 1) * getMatrixValue(B, 2, 1)
                // + getMatrixValue(A, i, 2) * getMatrixValue(B, 2, 2)
            }

            // setMatrixValue(res, i, g, sum);
        }
    }

    return res;
}

struct Matrix *addMatrix(struct Matrix *A, struct Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols)
    {
        printf("[add] incompatible size error");
        return NULL;
    }

    int rows = A->rows;
    int cols = A->cols;
    float *data = calloc(rows * cols, sizeof(float));

    for (int i = 0; i < rows; i++)
    {
        for (int g = 0; g < cols; g++)
        {
            data[i * cols + g] = getMatrixValue(A, i, g) + getMatrixValue(B, i, g);
        }
    }

    return createMatrix(rows, cols, data, false, A, B, '+');
}



struct Matrix *MSEMatrix(struct Matrix *predicted, struct Matrix *actual) {

    float* data = calloc(1, sizeof(float));
    struct Matrix *MSE = createMatrix(1, 1, data, false, predicted, actual, 'E');


    for (int i = 0; i < predicted->rows * predicted->cols; i++) {
        MSE->data[0] += (predicted->data[i] - actual->data[i]) * (predicted->data[i] - actual->data[i]);
    }

    MSE->data[0] /= predicted->rows * predicted->cols;

    return MSE;
}

// we dont need transpose as a operation that tracks gradients, since its only used internally for the backward pass of matmul.
// struct Matrix *transposeMatrix(struct Matrix *input)
// {
//     float *newData = malloc(sizeof(float) * input->cols * input->rows);
//     struct Matrix *mat = createMatrix(input->cols, input->rows, newData, false, input, NULL, 'T');

//     for (int i = 0; i < input->rows; i++)
//     {
//         for (int g = 0; g < input->cols; g++)
//         {
//             newData[g * input->rows + i] = getMatrixValue(input, i, g);
//         }
//     }
//     return mat;
// }

void printMatrix(struct Matrix *mat)
{
    for (int i = 0; i < mat->rows; i++)
    {
        for (int g = 0; g < mat->cols; g++)
        {
            printf("%f ", getMatrixValue(mat, i, g));
        }
        printf("\n");
    }
}