#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "graph.h"
#include "helper.h"

int main(void)
{

    srand(time(NULL));

    //setup layers
    //input should be 10 rows x 1 col (1 hot encoded of input 0-9)
    struct Matrix* embedding_table = createRandMatrix(10, 9, true); // 10 x 9 (vocab x dim)
    struct Matrix* layer2 = createRandMatrix(9, 3, true);           // 1x9 @ 9x3 -> 1x3
    struct Matrix* layer3 = createRandMatrix(1, 3, true);          // bias matches 1x3 activations
    struct Matrix* layer4 = createRandMatrix(3, 10, true);          // 1x3 @ 3x10 -> 1x10 logits
    // targets: 1x10 row (same layout as logits) for 10 classes

    
    for (int i = 0; i < 1000; i++)
    {

        // create random x, y input pair
        float* x = calloc(10, sizeof(float));
        int x_index = rand() % 10;
        x[x_index] = 1;

        float* y = calloc(10, sizeof(float));
        y[(3*x_index+5)%10] = 1;

        //create 1 hot encoded input/output
        struct Matrix *input = createMatrix(1, 10, x, false, NULL, NULL, 'l');
        struct Matrix *output = createMatrix(1, 10, y, false, NULL, NULL, 'l');


        // forward pass: row-vector batch (1x10 one-hot @ 10x9 -> 1x9 -> ... -> 1x10 logits)
        struct Matrix *embedding_table_output = multiplyMatrix(input, embedding_table);
        struct Matrix *layer2_output = multiplyMatrix(embedding_table_output, layer2);
        struct Matrix *layer3_output = addMatrix(layer2_output, layer3);
        struct Matrix *layer4_output = multiplyMatrix(layer3_output, layer4);
        struct Matrix *MSE_loss = MSEMatrix(layer4_output, output);

        //print loss
        printf("Iteration %i Loss: %f\n", i, MSE_loss->data[0]);

        //backward pass
        struct Matrix **topologicalArray = calloc(1000, sizeof(struct Matrix *));
        int size = 0;
        getTopo(MSE_loss, topologicalArray, &size);
        backward(topologicalArray, size);
        //update weights
        updateWeights(topologicalArray, size, .001);

        //clean up
        zeroGrad(topologicalArray, size);
        resetVisited(topologicalArray, size);

    }

    return 0;
}