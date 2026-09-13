#include "value-autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//  struct Value **topologicalArray is a pointer to a POIINTER OF STRUCT VALUES
// this means we are pointing to where we have an array of struct value pointers
void getTopo(struct Value *head, struct Value **topologicalArray, int *size)
{
    if (head == NULL || head->isVisited == 1)
    {
        return;
    }

    head->isVisited = true;
    // get children first
    getTopo(head->child1, topologicalArray, size);
    getTopo(head->child2, topologicalArray, size);

    // add this one
    topologicalArray[*size] = head;
    *size += 1;
}
// sets the children gradients!
void backward(struct Value **topologicalArray, int size)
{
    // we have topoligcal array in format: child leaf, child all the way to --> head
    topologicalArray[size - 1]->grad = 1; // initial gradient has to be set to 1. D-out/D-out = 1

    for (int i = size - 1; i >= 0; i--)
    {

        struct Value *currentNode = topologicalArray[i];

        switch (currentNode->op)
        {
        case 'l': // leaf
            break;

        case '+':
            currentNode->child1->grad += currentNode->grad;
            currentNode->child2->grad += currentNode->grad;
            break;
        case '*':
            currentNode->child1->grad += (currentNode->child2->data) * currentNode->grad;
            currentNode->child2->grad += (currentNode->child1->data) * currentNode->grad;
            break;
        case 't': // tanh
            // currentNode->child1->grad += (1 - (tanh(currentNode->child1->data))^2 * * currentNode->grad;
            //  tanh backward is 1-tanh^2(x)
            //  we already have tanh(x) from the forward pass, so we can simplify this
            currentNode->child1->grad += (1 - currentNode->data * currentNode->data) * currentNode->grad;
            break;

        default:
            printf("error! default case hit");
            break;
        }
    }
}

void freeGraph(struct Value **topologicalArray, int size)
{
    // we want to free the head LAST because then we loose referneces.
    for (int i = 0; i < size; i++)
    {
        free(topologicalArray[i]); // free expects a pointer. since we have double pointer, [] dereferences 1 pointer so we doing right.
    }

    free(topologicalArray);
}

void zeroGrad(struct Value **topologicalArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        topologicalArray[i]->grad = 0;
        topologicalArray[i]->isVisited = 0; // might need this if training runs this again
    }
}

struct Value *createLeafValue(float data, bool updatable)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));
    newValPointer->data = data;
    newValPointer->isVisited = 0;
    newValPointer->grad = 0.0;
    newValPointer->isLeaf = 1;
    newValPointer->child1 = NULL;
    newValPointer->child2 = NULL;
    newValPointer->op = 'l';
    newValPointer->isUpdatable = updatable;

    return newValPointer;
}

struct Value *createRandomLeafValue(bool updatable)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));
    float randVal = ((float)(rand() % 100) + 1) / 100.0f;

    newValPointer->data = randVal;
    newValPointer->isVisited = 0;
    newValPointer->grad = 0.0;
    newValPointer->isLeaf = 1;
    newValPointer->child1 = NULL;
    newValPointer->child2 = NULL;
    newValPointer->op = 'l';
    newValPointer->isUpdatable = updatable;

    return newValPointer;
}

struct Value *
addValue(struct Value *val1, struct Value *val2)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));

    newValPointer->data = val1->data + val2->data;
    newValPointer->grad = 0.0;
    newValPointer->isVisited = 0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = val2;
    newValPointer->op = '+';

    newValPointer->isUpdatable = false;

    return newValPointer;
}
struct Value *multiplyValue(struct Value *val1, struct Value *val2)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));

    newValPointer->data = (val1->data) * (val2->data);
    newValPointer->grad = 0.0;
    newValPointer->isVisited = 0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = val2;
    newValPointer->op = '*';

    newValPointer->isUpdatable = false;

    return newValPointer;
}

struct Value *tanhValue(struct Value *val1)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));

    newValPointer->data = tanh(val1->data);
    newValPointer->grad = 0.0;
    newValPointer->isVisited = 0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = NULL;
    newValPointer->op = 't'; // tanh

    newValPointer->isUpdatable = false;

    return newValPointer;
}

void printValue(struct Value *val)
{
    printf("\nValue %p\nData: %f\nGrad: %f\nisLeaf: %d\nChild1 Pointer: %p\nChild2 Pointer: %p\nOperation: %c\nIs Updatable: %d\n",
           (void *)val,
           val->data,
           val->grad,
           val->isLeaf,
           (void *)val->child1,
           (void *)val->child2,
           val->op,
           val->isUpdatable);
}

void printTree(struct Value *head)
{
    printValue(head);

    if (head->child1 != NULL)
    {
        printTree(head->child1);
    }
    if (head->child2 != NULL)
    {
        printTree(head->child2);
    }
}