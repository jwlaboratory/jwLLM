#include "value-autograd.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    srand(time(NULL));

    // setup layers
    struct Embedding_Table *emb_table = create_embedding_table(10, 9); // layer 1: embedding table size size 10x9
    struct Neuron *neuron1 = createNeuron(3);                          // l2
    struct Neuron *neuron2 = createNeuron(3);
    struct Neuron *neuron3 = createNeuron(3);
    struct Neuron *neruon4 = createNeuron(3); // l3/4

    for (int i = 0; i < 5000; i++)
    {
        // training y = 3x + 5 for [0-9] for x
        int x = rand() % 10; // in
        // int x = 3;
        float y = (3 * x + 5); // out

        struct Tensor *curTensor = emb_table->table[x]; // plug in x
        // layer 2:
        struct Value *post_n1 = forwardNeuron(neuron1, get_sub_array_from_tensor(curTensor, 0, 3));
        struct Value *post_n2 = forwardNeuron(neuron2, get_sub_array_from_tensor(curTensor, 3, 6));
        // layer 3 tanH on neuron2:
        struct Value *post_tanh_n2 = tanhValue(post_n2);
        // layer 2 on neuron3
        struct Value *post_n3 = forwardNeuron(neuron3, get_sub_array_from_tensor(curTensor, 6, 9));
        // layer 4
        struct Value **arr_from_layer3 = calloc(3, sizeof(struct Value *));
        arr_from_layer3[0] = post_n1;
        arr_from_layer3[1] = post_tanh_n2;
        arr_from_layer3[2] = post_n3;
        struct Value *out = forwardNeuron(neruon4, arr_from_layer3);
        // loss MSE
        struct Value *actualVal = createLeafValue(-1 * y, false);
        struct Value *loss = addValue(out, actualVal);
        struct Value *squared_loss = multiplyValue(loss, loss);

        // build topologicalArray here
        struct Value **topologicalArray = calloc(1000, sizeof(struct Value *)); // TODO: we need to know how much mem to allocate, fix
        int size = 0;
        getTopo(squared_loss, topologicalArray, &size);

        // backward pass
        backward(topologicalArray, size);
        // printValue(squared_loss);
        printf("Iteration %i Loss: %f X: %d Actual Y: %f Predicted Y: %f\n", i, squared_loss->data, x, y, out->data);
        // printTree(squared_loss);

        // update weights using gradients
        updateWeights(topologicalArray, size, .001);

        // clean up, reset gradients and free graph
        zeroGrad(topologicalArray, size);
    }

    // freeGraph(topologicalArray, size);

    // test
    // struct Embedding_Table *emb_tabletest = create_embedding_table(10, 9);
    // struct Tensor *test = emb_tabletest->table[x]; // plug in x
    // struct Neuron *testn = createNeuron(3);
    // struct Value *outtest = forwardNeuron(testn, get_sub_array_from_tensor(test, 0, 3));
    // printTree(outtest);
}

// todo:

// 4. fix memory issues (post on x)
//// 1. fix topological array line 41

// 5. after this training loop remove notion of neurons switch to matrix multiplication
// 6. train MLP as per bengio et all

// 7. train RNN
// 8. train CNN

// resume add: Autograd engine complete, create MLP, RNN, CNN using it