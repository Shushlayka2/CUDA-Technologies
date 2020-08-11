#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define BLOCK_SIZE 256
#define ArraySize 1000
#define AbsMaxVal 10

void generate_random_arr(float* A);

__constant__ float A[ArraySize];
__constant__ float B[ArraySize];

__global__ void scalMult(float* C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ArraySize)
	{
		float sum = 0.0f;
		if (threadIdx.x == 0)
		{
			sum = 0.0;
			for (int j = 0; j < blockDim.x; j++)
				sum += A[j] * B[j];

			C[blockIdx.x] = sum;
		}
	}
}

int main()
{
	int GRID_SIZE = ArraySize / BLOCK_SIZE + (ArraySize % BLOCK_SIZE != 0 ? 1 : 0);
	srand(time(NULL));

	float *C, *Temp;
	float* d_C;
	C = (float*)malloc(sizeof(float) * GRID_SIZE);
	Temp = (float*)malloc(sizeof(float) * ArraySize);
	generate_random_arr(Temp);
	cudaMemcpyToSymbol(A, Temp, sizeof(float) * ArraySize);
	generate_random_arr(Temp);
	cudaMemcpyToSymbol(B, Temp, sizeof(float) * ArraySize);
	cudaMalloc((void**)&d_C, sizeof(float) * GRID_SIZE);

	float KernelTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	scalMult << <GRID_SIZE, BLOCK_SIZE >> > (d_C);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&KernelTime, start, stop);

	cudaMemcpy(C, d_C, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
		result += C[i];
	printf("Result: %f\nElapsedTime: %f", result, KernelTime);

	free(C);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}

void generate_random_arr(float* A)
{
	for (int i = 0; i < ArraySize; i++)
	{
		A[i] = (rand() / (float)RAND_MAX) * (AbsMaxVal * 2 + 1) + -1 * AbsMaxVal;
		printf("%f ", A[i]);
	}
	printf("\n");
}