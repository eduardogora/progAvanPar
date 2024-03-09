
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void matrizTransG(int* matA, int* matB, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows  && col < cols) {
        int idOriginal = (row * cols) + col;
        int idTrans = (col * rows) + row;

        matB[idTrans] = matA[idOriginal];
    }
}

void printMatriz(int* matriz, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%i, ", matriz[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrizTransD(int* a, int* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            b[rows * j + i] = a[cols * i + j];
        }
    }
}

int main()
{

    //Datos 
    int tempA[] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

    int N = 3, M = 4;
    int* matA, *matB;

    int size = N * M * sizeof(int);

    matA = (int*)malloc(size);
    matB = (int*)malloc(size);

    matA = &tempA[0];

    //CPU
    printf("\n ORIGINAL: \n");
    printMatriz(matA, N, M);
    matrizTransD(matA, matB, N, M);
    printf("\n CPU: \n");
    printMatriz(matB, M, N);

    //GPU
    dim3 blockSize(4, 4, 1);
    dim3 gridSize(1, 1, 1);

    int* a_cpu;
    int* b_cpu;


    int* a_device;
    int* b_device;

    a_cpu = (int*)malloc(size);
    b_cpu = (int*)malloc(size);

    a_cpu = &tempA[0];

    cudaMalloc((void**)&a_device, size);
    cudaMalloc((void**)&b_device, size);

    //transfer to GPU memory
    cudaMemcpy(a_device, a_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, size, cudaMemcpyHostToDevice);

    //kernel launch
    matrizTransG << <gridSize, blockSize >> > (a_device, b_device, N, M);


    //transfer to CPU host memory from GPU device //source, from, size
    cudaMemcpy(a_cpu, a_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, size, cudaMemcpyDeviceToHost);

    printf("\n GPU: \n");
    printMatriz(b_cpu, M, N);

    //limpieza
    cudaDeviceReset();
    cudaFree(a_device);
    cudaFree(b_device);



    cudaDeviceSynchronize();
    return 0;
}
