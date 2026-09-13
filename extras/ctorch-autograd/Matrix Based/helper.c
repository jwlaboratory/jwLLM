#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void backward_helper_matmul(float* a, int ar, int ac, float* b, int br, int bc, float* c)
{
    

    for (int i = 0; i < ar; i++)
    {
        for (int g = 0; g < bc; g++)
        {

            for (int q = 0; q < ac; q++)
            {
                c[i * bc + g] += a[i*ac+q] * b[q*bc+g]; 

            }

        }
    }
}

void backward_helper_transpose(float* a, int ar, int ac, float*b) {
    for (int i = 0; i < ar; i++)
    {
        for (int g = 0; g < ac; g++)
        {
            b[g * ar + i] = a[i * ac + g];
        }
    }
}