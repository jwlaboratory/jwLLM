

#include "value-autograd.h"
#include <stdio.h>
#include <stdlib.h>

struct Neuron *createNeuron(int numInputs)
{

    struct Neuron *neuron = malloc(sizeof(struct Neuron));
    neuron->weights = calloc(numInputs, sizeof(struct Value *));

    for (int i = 0; i < numInputs; i++)
    {
        neuron->weights[i] = createRandomLeafValue(true);
    }
    neuron->num_inputs = numInputs;
    neuron->bias = createLeafValue(0, true);

    return neuron;
}

struct Value *forwardNeuron(struct Neuron *neuron, struct Value **inputs)
{
    int size = neuron->num_inputs;
    if (size != neuron->num_inputs)
    {
        return NULL; // err
    }

    struct Value *curFinalSumPointer = neuron->bias; // includes bias already

    for (int i = 0; i < size; i++)
    {
        struct Value *scaled = multiplyValue(neuron->weights[i], inputs[i]);
        curFinalSumPointer = addValue(curFinalSumPointer, scaled);
    }

    neuron->head_out = curFinalSumPointer;
    return curFinalSumPointer;
}

// assumes backward() already ran
// potentially not used
// void updateNeuronParams(struct Neuron *neuron, float learning_rate)
// {
//     for (int i = 0; i < neuron->num_inputs; i++)
//     {
//         struct Value *curWeight = neuron->weights[i];
//         // update as per learning rate
//         curWeight->data -= curWeight->grad * learning_rate;
//     }

//     // update bias aswell
//     neuron->bias->data -= neuron->bias->grad * learning_rate;
// }

// assumes backward() already ran
void updateWeights(struct Value **topologicalArr, int size, float learning_rate)
{
    for (int i = 0; i < size; i++)
    {
        if (topologicalArr[i]->isUpdatable)
        {
            topologicalArr[i]->data -= topologicalArr[i]->grad * learning_rate;
        }
    }
}