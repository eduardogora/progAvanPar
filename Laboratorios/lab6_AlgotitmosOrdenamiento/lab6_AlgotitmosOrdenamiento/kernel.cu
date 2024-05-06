
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*----------------------------CPU-------------------------*/
// Función para imprimir un array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Función para llenar un arreglo con números aleatorios
void fillArrayWithRandomNumbers(int arr[], int n, int max_value) {
    // Semilla para la generación de números aleatorios
    srand(time(nullptr));

    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % (max_value + 1); // Genera un número aleatorio entre 0 y max_value
    }
}

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // Tomamos el último elemento como pivote
    int i = low - 1; // Índice del elemento más pequeño

    for (int j = low; j < high; j++) {
        // Si el elemento actual es menor o igual al pivote
        if (arr[j] <= pivot) {
            i++; // Incrementamos el índice del elemento más pequeño
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1; // Retornamos la posición del pivote
}


//Merge srot y BitonicSort son los mas paralelizables
//GenerateReducable -> yes , en projects -> cuda
//global siempre te pide que sea void, device puede retornar algun valor

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        // Obtenemos la posición del pivote
        int pi = partition(arr, low, high);

        // Ordenamos los elementos antes y después del pivote
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

/*----------------------------GPU-------------------------*/
__device__ int partition_GPU(int* arr, int low, int high) {
    int pivot = arr[high]; // Tomar el último elemento como pivote
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            // Intercambiar arr[i] y arr[j]
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    // Intercambiar arr[i + 1] y arr[high] (pivote)
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

__global__ void quicksort_GPU(int* arr, int left, int right) {
    int* stack = new int[right - left + 1]; // Pila para almacenar los límites izquierdo y derecho
    int top = -1; // Inicializar la cima de la pila

    // Empujar los límites izquierdo y derecho iniciales en la pila
    stack[++top] = left;
    stack[++top] = right;

    // Iterar hasta que la pila esté vacía
    while (top >= 0) {
        // Desapilar los límites izquierdo y derecho actuales
        right = stack[top--];
        left = stack[top--];

        // Hacer la partición y obtener el índice del pivote
        int pivotIndex = partition_GPU(arr, left, right);

        // Si hay elementos en el lado izquierdo del pivote, empujar sus límites en la pila
        if (pivotIndex - 1 > left) {
            stack[++top] = left;
            stack[++top] = pivotIndex - 1;
        }

        // Si hay elementos en el lado derecho del pivote, empujar sus límites en la pila
        if (pivotIndex + 1 < right) {
            stack[++top] = pivotIndex + 1;
            stack[++top] = right;
        }
    }

    delete[] stack; // Liberar la memoria de la pila
}




int main()
{
    const int numVal = 200000;
    int arr[numVal];
    int n = sizeof(arr) / sizeof(arr[0]);

    fillArrayWithRandomNumbers(arr, n, 100000);

    //cout << "Array original:\n";
    //printArray(arr, n);

    /*--------------------QUICKSORT CPU----------------------*/
    // Medir el tiempo de ejecución
    auto start = high_resolution_clock::now(); // Tiempo de inicio

    quicksort(arr, 0, n - 1);

    auto stop = high_resolution_clock::now(); // Tiempo de fin
    auto duration = duration_cast<milliseconds>(stop - start); // Duración en milisegundos

    cout << "Tiempo de ejecucion quicksort CPU: " << duration.count() << " milisegundos" << endl;



    /*-------------------QUICKSORT GPU-----------------------*/
    // Copiar el arreglo a la memoria de la GPU
    int arrQSGPU[numVal];
    fillArrayWithRandomNumbers(arrQSGPU, n, 100000);

    int* d_arr;
    cudaMalloc(&d_arr, numVal * sizeof(int));
    cudaMemcpy(d_arr, arrQSGPU, numVal * sizeof(int), cudaMemcpyHostToDevice);

    // Llamar al kernel de CUDA para Quicksort
    quicksort_GPU << <1, 1 >> > (d_arr, 0, numVal - 1);
    cudaDeviceSynchronize(); // Esperar a que todos los hilos terminen

    // Copiar el arreglo ordenado de vuelta a la CPU
    cudaMemcpy(arrQSGPU, d_arr, numVal * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir el tiempo de ejecución


    cout << "Array ordenado con Quicksort:\n";
    //printArray(arr, n);

    return 0;
}
