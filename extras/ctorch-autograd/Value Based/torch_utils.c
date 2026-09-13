#include "value-autograd.h"
#include <stdlib.h>
#include <string.h>

// inclusive exclusive, ie: [start, end)
struct Value **get_sub_array_from_tensor(struct Tensor *tensor, int startIndex, int endIndex)
{
    struct Value **new_ar = malloc(sizeof(struct Value *) * (endIndex - startIndex));

    memcpy(new_ar, tensor->array + startIndex, (endIndex - startIndex) * sizeof(struct Value *));

    return new_ar;
}

struct Tensor *createTensor(int width, bool createRandVals, bool updatable)
{
    struct Tensor *tensor = malloc(sizeof(struct Tensor));
    tensor->width = width;
    tensor->array = calloc(width, sizeof(struct Value *));
    // width amount of pointers in the array

    for (int i = 0; i < width; i++)
    {
        if (createRandVals)
        {
            tensor->array[i] = createRandomLeafValue(updatable);
        }
        else
        {
            tensor->array[i] = createLeafValue(0, updatable);
        }
    }

    return tensor;
}

struct Embedding_Table *create_embedding_table(int length, int width)
{
    struct Embedding_Table *emb_table = malloc(sizeof(struct Embedding_Table));
    emb_table->size = length;
    emb_table->table = calloc(length, sizeof(struct Tensor *));
    for (int i = 0; i < length; i++)
    {
        struct Tensor *row = createTensor(width, true, true);
        emb_table->table[i] = row;
    }
    return emb_table;
}

struct Value *loss_function_MSE(struct Value *predicted, float actual)
{
    // I know for lik ea next token predictor tat u can use neg log likelihood to maximize predicted probability  but htat assumes the output of ur network is like 10 values that each correspond to the probablitiy of one value each
    // In the context of this loss function diagram, "Regression" refers to predicting continuous numerical values, like fitting a line to data such as y = 3x + 5.

    struct Value *actualVal = createLeafValue(-1 * actual, false);
    struct Value *loss = addValue(predicted, actualVal);
    struct Value *squared_loss = multiplyValue(loss, loss);

    return squared_loss;
}