#include <stdbool.h>
#pragma once

struct Matrix
{
    int rows;
    int cols;

    float *data;
    float *grads;

    bool isVisited;   // for topological sort
    bool isUpdatable; // should we update this as a weight / bias? or is it itermediate value

    // If it's not a leaf, we fill these in.
    struct Matrix *child1;
    struct Matrix *child2;
    char op;
};

struct Matrix *createMatrix(int rows, int cols, float *data, bool isUpdatable, struct Matrix *child1, struct Matrix *child2, char op);
struct Matrix *createRandMatrix(int rows, int cols, bool isUpdatable);
void printMatrix(struct Matrix *mat);

// helper
float getMatrixValue(struct Matrix *mat, int row, int col);
void setMatrixValue(struct Matrix *mat, int row, int col, float data);

// operations
struct Matrix *multiplyMatrix(struct Matrix *A, struct Matrix *B);
struct Matrix *addMatrix(struct Matrix *A, struct Matrix *B);
struct Matrix *MSEMatrix(struct Matrix *predicted, struct Matrix *actual);
// struct Matrix *transposeMatrix(struct Matrix *input);

