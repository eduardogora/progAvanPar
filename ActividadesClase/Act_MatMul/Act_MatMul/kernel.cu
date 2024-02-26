
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>


__global__ void multMatrizGPU(int* a, int* b, int* c, int width, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int total = 0;

    for (int k = 0; k < width; k++) {
        total += a[row * width + k] * b[k * width + col];
        //printf("total: %i\n", total);
    }
    c[row * width + col] = total;


}

void multMatriz(int* matA, int* matB, int* matC, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int total = 0;
            for (int k = 0; k < m; k++) {
                total += matA[i * m + k] * matB[k * n + j];
            }
            matC[i * n + j] = total;
        }
    }
}

void prodCruz(int* matA, int* matB, int* matC) {

}

void printMatriz(int* matriz, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%i, ", matriz[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fillMatriz(int* matriz, int n, int m, int num) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matriz[i * n + j] = num;
        }
    }
}

int main()
{
    //Declaracion de Matrices
    int* matA, *matB, *matC, *matPunto;
    int N = 4, M = 4;

    int size = N * M * sizeof(int);

    matA = (int*)malloc(size);
    matB = (int*)malloc(size);
    matC = (int*)malloc(size);
    matPunto = (int*)malloc(size);

    //FIll matrix
    fillMatriz(matA, N, M, 2);
    fillMatriz(matB, N, M, 1);

    //Print Matrixes
    printMatriz(matA, N, M);
    printMatriz(matB, N, M);

    //producto cruz
    multMatriz(matA, matB, matC, N, M);
    printMatriz(matC, N, M);

    //Declaramos el tamano del grid y bloque
    int BLOCK_SIZE = 32;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize(ceil(N / BLOCK_SIZE), ceil(N / BLOCK_SIZE), 1);

    //Inicializamos punteros (informacion y variables) dentro del CPU
    int* c_cpu;
    int* a_cpu;
    int* b_cpu;

    //Inicializamos punteros (informacion y variables) dentro del GPU
    int* c_device;
    int* a_device;
    int* b_device;

    //Reservamos memoria en el GPU
    c_cpu = (int*)malloc(size);
    a_cpu = (int*)malloc(size);
    b_cpu = (int*)malloc(size);

    //Memory allocation
    cudaMalloc((void**)&c_device, size);
    cudaMalloc((void**)&a_device, size);
    cudaMalloc((void**)&b_device, size);

    //transfer to GPU memory
    cudaMemcpy(c_device, c_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, size, cudaMemcpyHostToDevice);

    //kernel launch
    multMatrizGPU << <gridSize, blockSize >> > (a_device, b_device, c_device, 4, N, M);

    //transfer to CPU host memory from GPU device
    cudaMemcpy(c_cpu, c_device, size, cudaMemcpyDeviceToHost); //source, from, size
    cudaMemcpy(a_cpu, a_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, size, cudaMemcpyDeviceToHost);

    //printMatrizCuda
    printMatriz(c_cpu, N, M);

    //limpieza
    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);



    cudaDeviceSynchronize();
    return 0;
}
