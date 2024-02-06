
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void globalId()
{
    int hiloX = threadIdx.x;
    int hiloY = threadIdx.y;

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int dimX = blockDim.x;
    int dimY = blockDim.y;

    int globalIDx = blockX * dimX + hiloX;
    int globalIDy = blockY * dimY + hiloY;

    int gId = globalIDy * blockDim.x * gridDim.x + globalIDx;

    //printf("gID: %d \n", gId);
    printf("%d \n", gId);
}

int main()
{
    dim3 blockSize(4, 2);

    dim3 gridSize(2, 2);

    globalId << <gridSize, blockSize >> > ();
    cudaDeviceSynchronize();
    return 0;
}

