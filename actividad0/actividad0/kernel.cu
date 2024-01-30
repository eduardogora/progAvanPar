#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_hello_cuda() {
	int i = threadIdx.x;
	printf("[PRINT] ThreadId.x: %d \n", i);
}

int main() {
	print_hello_cuda << <1, 8 >> > ();
	return 0;
}