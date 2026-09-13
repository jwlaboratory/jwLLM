#include <stdbool.h>
#pragma once

// Core scalar value in the autograd graph.
struct Value
{
    float data;
    float grad; // filled during backward pass
    bool isLeaf;
    bool isVisited;   // for topological sort
    bool isUpdatable; // should we update this as a weight / bias? or is it itermediate value

    // If it's not a leaf, we fill these in.
    struct Value *child1;
    struct Value *child2;
    char op; // '+', '*', etc. '\0' for leaf
};

struct Value *createLeafValue(float data, bool updatable);
struct Value *createRandomLeafValue(bool updatable);
struct Value *addValue(struct Value *val1, struct Value *val2);
struct Value *multiplyValue(struct Value *val1, struct Value *val2);
struct Value *tanhValue(struct Value *val1);

void printValue(struct Value *val);
void printTree(struct Value *head);

void getTopo(struct Value *head, struct Value **topologicalArray, int *size);
void backward(struct Value **topologicalArray, int size);
void freeGraph(struct Value **topologicalArray, int size);
void zeroGrad(struct Value **topologicalArray, int size);

// for neural.c
struct Neuron
{
    struct Value **weights;
    struct Value *bias;
    struct Value *head_out; // after doing forward pass gets created and saved here
    int num_inputs;
};

struct Neuron *createNeuron(int numInputs);
struct Value *forwardNeuron(struct Neuron *neuron, struct Value **inputs);
// void updateNeuronParams(struct Neuron *neuron, float learning_rate);
void updateWeights(struct Value **topologicalArr, int size, float learning_rate);

// for torch_utils.c
struct Tensor
{
    int width;
    struct Value **array; // pointer to value pointer
};

struct Embedding_Table
{
    // we want to know size, a pointer to an array of vectors
    int size;
    struct Tensor **table; // pointer to an array of tensor
};

struct Tensor *createTensor(int width, bool createRandVals, bool updatable);
struct Value **get_sub_array_from_tensor(struct Tensor *tensor, int startIndex, int endIndex);
struct Embedding_Table *create_embedding_table(int length, int width);
struct Value *loss_function_MSE(struct Value *predicted, float actual);