#include <stdbool.h>
#pragma once


void backward_helper_matmul(float* a, int ar, int ac, float* b, int br, int bc, float* c);
void backward_helper_transpose(float* a, int ar, int ac, float*b);