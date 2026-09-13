#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "graph.h"
#include "helper.h"


int main() {
    srand(time(NULL));

    int N = 512;

    struct Matrix* A = createRandMatrix(N, N, true);
    struct Matrix* B = createRandMatrix(N, N, true);

    // warmup
    multiplyMatrix(A, B);

    double start = clock();
    for (int i = 0; i < 50; i++) {
        multiplyMatrix(A, B);
    }
    double end = clock();

    printf("Time per matmul: %f ms\n", (end - start) / 50.0);
    printf("floating point operations per second: %f\n", (N * N * N * 2) / ((end - start) / 50.0));
}