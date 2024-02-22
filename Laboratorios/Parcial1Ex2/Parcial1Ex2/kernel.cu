
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;


__global__ void orBit(int* a, int* b, int* c, int size)
{
    int hiloX = threadIdx.x;
    int hiloY = threadIdx.y;

    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int dimX = blockDim.x;
    int dimY = blockDim.y;

    int globalIDx = blockX * dimX + hiloX;
    int globalIDy = blockY * dimY + hiloY;

    int gId = (globalIDy * blockDim.x * gridDim.x) + globalIDx;

    if (gId < size) {
        c[gId] = a[gId] | b[gId];
    }
    
}

int main()
{
    //

    //Declaramos Variables
    const int size_in_bytes = 10000;
    //printf("%d", sizeY);

    //Declaramos el tamano del grid y bloque
    //La distribucion optima es de 64 hilos por bloque ya que 
    //tenemos un maximo de 1024 hilos y 16 bloques
    //por lo que 1024/16 = 64 hilos por bloque
    //por lo que 16 bloques por 46 SM sera nuestro
    //tamanio de grid. N-Size vale 64*16*46 = 47,104

    //multiplicas 1024*16*46 revisando cuantos hilos necesitas revisando limite de hilos por bloque

    /*--------------------------------Respuesta---------------------------------------------------------------------------------------------|
    | Dado que contamos con un total de 754,687 hilos (obtenidos de multiplicar nuestro maximo de hilos, por la cantidad de bloques y de SM |
    | Por lo que la cantidad de bloques necesarios sería dividir la cantidad de hilos, entre la cantidad maxima de hilos por bloque         |
    | dado que buscamos la mejor optimización. Lo que nos dá una cantidad de 737 bloques.  Con esto obtendriamos la mejor                   |
    | configuración con estas condiciones.                                                                                                  |
    ---------------------------------------------------------------------------------------------------------------------------------------*/

    dim3 blockSize(1024, 1, 1);
    dim3 gridSize(737, 1, 1);

    unsigned int a_cpu[size_in_bytes];
    unsigned int b_cpu[size_in_bytes];
    unsigned int c_cpu[size_in_bytes];

    for (int i = 0; i < size_in_bytes; i++) {
        a_cpu[i] = 0;
        b_cpu[i] = 1;
    }

    //Inicializamos punteros (informacion y variables) dentro del GPU
    int* a_device;
    int* b_device;
    int* c_device;

    //Reservamos memoria en el GPU
    const int dataCount = size_in_bytes;
    const int data_size = dataCount * sizeof(unsigned int);

    //Memory allocation
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);
    cudaMalloc((void**)&c_device, data_size);

    //transfer to GPU memory
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);

    //kernel launch
    orBit << <gridSize, blockSize >> > (a_device, b_device, c_device, size_in_bytes);

    //transfer to CPU host memory from GPU device
    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost); //source, from, size
    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);


    //Work
    printf("\n Vector Resultante: \n");
    for (int i = 0; i < size_in_bytes; ++i) {
        printf("%d \n ", c_cpu[i]);
    }


    //limpieza
    cudaDeviceReset();
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);



    cudaDeviceSynchronize();
    return 0;
}
