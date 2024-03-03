
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>



void printMatriz(int* matriz, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%i, ", matriz[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void dConvolucion(int* matA, int* matB, int* filtro, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int id = (row * cols) + col;
    int total = 0;

    //Primera fila
    if (row == 0) {
        if (col == 0) {
            //Segunda Fila
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            //Tercer fila
            total += matA[(row + 1) * cols + (col)] * filtro[7];
            total += matA[(row + 1) * cols + (col + 1)] * filtro[8];

            matB[id] = total;
        }
        else if (col == cols - 1) {
            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];

            //Tercer fila
            total += matA[(row + 1) * cols + (col - 1)] * filtro[6];
            total += matA[(row + 1) * cols + (col)] * filtro[7];

            matB[id] = total;
        }
        else {
            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            //Tercer fila
            total += matA[(row + 1) * cols + (col - 1)] * filtro[6];
            total += matA[(row + 1) * cols + (col)] * filtro[7];
            total += matA[(row + 1) * cols + (col + 1)] * filtro[8];

            matB[id] = total;
        }
    }
    //Ultima fila
    else if (row == rows - 1) {
        if (col == 0) {
            total += matA[(row - 1) * cols + (col)] * filtro[1];
            total += matA[(row - 1) * cols + (col + 1)] * filtro[2];

            //Segunda Fila
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            matB[id] = total;
        }
        else if (col == cols - 1) {
            total += matA[(row - 1) * cols + (col - 1)] * filtro[0];
            total += matA[(row - 1) * cols + (col)] * filtro[1];

            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];

            //Tercer fila

            matB[id] = total;
        }
        else {
            total += matA[(row - 1) * cols + (col - 1)] * filtro[0];
            total += matA[(row - 1) * cols + (col)] * filtro[1];
            total += matA[(row - 1) * cols + (col + 1)] * filtro[2];

            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            matB[id] = total;
        }
    }
    else {
        if (col == 0) {
            //Fila 1
            total += matA[(row - 1) * cols + (col)] * filtro[1];
            total += matA[(row - 1) * cols + (col + 1)] * filtro[2];

            //Segunda Fila
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            //Tercer fila
            total += matA[(row + 1) * cols + (col)] * filtro[7];
            total += matA[(row + 1) * cols + (col + 1)] * filtro[8];

            matB[id] = total;
        }
        else if (col == cols - 1) {
            total += matA[(row - 1) * cols + (col - 1)] * filtro[0];
            total += matA[(row - 1) * cols + (col)] * filtro[1];

            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];

            //Tercer fila
            total += matA[(row + 1) * cols + (col - 1)] * filtro[6];
            total += matA[(row + 1) * cols + (col)] * filtro[7];

            matB[id] = total;
        }
        else {
            total += matA[(row - 1) * cols + (col - 1)] * filtro[0];
            total += matA[(row - 1) * cols + (col)] * filtro[1];
            total += matA[(row - 1) * cols + (col + 1)] * filtro[2];

            //Segunda Fila
            total += matA[row * cols + (col - 1)] * filtro[3];
            total += matA[row * cols + (col)] * filtro[4];
            total += matA[row * cols + (col + 1)] * filtro[5];

            //Tercer fila
            total += matA[(row + 1) * cols + (col - 1)] * filtro[6];
            total += matA[(row + 1) * cols + (col)] * filtro[7];
            total += matA[(row + 1) * cols + (col + 1)] * filtro[8];

            matB[id] = total;
        }
    }
}

void cConvolucion(int* matA, int* matB, int* filtro, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int total = 0;
            int id = (i * cols) + j;

            //Fila 0
            if (i == 0) {
                //primer columna
                if (j == 0) {
                    //Segunda Fila
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j)] * filtro[7];
                    total += matA[(i + 1) * cols + (j + 1)] * filtro[8];

                    //Promediamos
                    matB[id] = total;
                }
                //Ultima columna
                else if(j == cols-1){
                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j - 1)] * filtro[6];
                    total += matA[(i + 1) * cols + (j)] * filtro[7];

                    matB[id] = total;
                }
                //Resto de columnas
                else {
                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j - 1)] * filtro[6];
                    total += matA[(i + 1) * cols + (j)] * filtro[7];
                    total += matA[(i + 1) * cols + (j + 1)] * filtro[8];

                    matB[id] = total;
                }
            }
            
            //Ultima fila
            else if (i == rows - 1) {
                //primer columna
                if (j == 0) {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j)] * filtro[1];
                    total += matA[(i - 1) * cols + (j + 1)] * filtro[2];

                    //Segunda Fila
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    //Promediamos
                    matB[i * cols + j] = total;
                }
                //Ultima columna
                else if (j == cols - 1) {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j - 1)] * filtro[0];
                    total += matA[(i - 1) * cols + (j)] * filtro[1];

                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];

                    //Promediamos
                    matB[id] = total; 
                }
                //Resto de columnas
                else {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j - 1)] * filtro[0];
                    total += matA[(i - 1) * cols + (j)] * filtro[1];
                    total += matA[(i - 1) * cols + (j + 1)] * filtro[2];

                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    matB[id] = total;
                }

            }
            else {
                //primer columna
                if (j == 0) {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j)] * filtro[1];
                    total += matA[(i - 1) * cols + (j + 1)] * filtro[2];

                    //Segunda Fila
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j)] * filtro[7];
                    total += matA[(i + 1) * cols + (j + 1)] * filtro[8];

                    matB[id] = total;
                }
                //Ultima columna
                else if (j == cols - 1) {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j - 1)] * filtro[0];
                    total += matA[(i - 1) * cols + (j)] * filtro[1];

                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j - 1)] * filtro[6];
                    total += matA[(i + 1) * cols + (j)] * filtro[7];

                    matB[id] = total;
                }
                //Resto de columnas
                else {
                    //Primer fila
                    total += matA[(i - 1) * cols + (j - 1)] * filtro[0];
                    total += matA[(i - 1) * cols + (j)] * filtro[1];
                    total += matA[(i - 1) * cols + (j + 1)] * filtro[2];

                    //Segunda Fila
                    total += matA[i * cols + (j - 1)] * filtro[3];
                    total += matA[i * cols + (j)] * filtro[4];
                    total += matA[i * cols + (j + 1)] * filtro[5];

                    //Tercer fila
                    total += matA[(i + 1) * cols + (j - 1)] * filtro[6];
                    total += matA[(i + 1) * cols + (j)] * filtro[7];
                    total += matA[(i + 1) * cols + (j + 1)] * filtro[8];

                    matB[id] = total;
                }
            }
        }
    }
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

    //Declaramos relojes
    clock_t cpuStart, cpuStop, gpuStart, gpuStop;
    
    //Declaracion de Matrices
    int* CmatA, *GmatA;
    int* CresA, *GresA;
    int N = 4, M = 5;
    int* filtroD;

    int filtro[3][3] = { {0,  1, 0}, 
                         {1, -4, 1}, 
                         {0,  1, 0}, 
                    };
    int matT[4][5] = {{1, 0, 1, 2, 2},
                      {1, 1, 2, 2, 3},
                      {1, 2, 2, 6, 3}, 
                      {1, 1, 2, 2, 3}, 
                      };
    
    int matZ[4][5] = {{0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0},
    };

    int size = N * M * sizeof(int);

    CmatA = (int*)malloc(size);
    CresA = (int*)malloc(size);

    CmatA = &matT[0][0];
    filtroD = &filtro[0][0];


    //Print Matrixes
    printMatriz(&matT[0][0], N, M);
    cpuStart = clock();
    cConvolucion(&matT[0][0], &matZ[0][0], &filtro[0][0], N, M);
    cpuStop = clock();
    printMatriz(&matZ[0][0], N, M);

    //Memory allocation
    cudaMalloc((void**)&GmatA, size);
    cudaMalloc((void**)&GresA, size);
    cudaMalloc((void**)&filtroD, size);

    //transfer to GPU memory
    cudaMemcpy(GmatA, CmatA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(GresA, CresA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(filtroD, filtro, size, cudaMemcpyHostToDevice);

    //kernel launch
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gpuStart = clock();
    dConvolucion << <numBlocks, threadsPerBlock >> > (GmatA, GresA, filtroD, N, M);
    gpuStop = clock();

    //transfer to CPU host memory from GPU device
    cudaMemcpy(CmatA, GmatA, size, cudaMemcpyDeviceToHost); //source, from, size
    cudaMemcpy(CresA, GresA, size, cudaMemcpyDeviceToHost);

    //printMatrizCuda
    printMatriz(CresA, N, M);

    //limpieza
    cudaDeviceReset();
    cudaFree(GmatA);
    cudaFree(GresA);



    cudaDeviceSynchronize();

    printf("Tiempo CPU: %d \n", (cpuStop - cpuStart) );
    printf("Tiempo GPU: %d \n", (gpuStop - gpuStart) );
    printf("Tiempo Ahora: %d", clock() );


    return 0;
}
