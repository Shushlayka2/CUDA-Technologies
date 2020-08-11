#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h> 
#include <malloc.h>

using namespace std;

#define ArrAmoun 20000
#define BlockSize 32
#define ValuesSize 1000
#define SumBlockSize 16

__global__ void multiply(float* arr1, float* arr2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ArrAmoun)
	{
		arr1[i] = __fmul_rn(arr1[i], arr2[i]);
	}
}

__global__ void sum(float* arr, float* res_arr, int size, int iterations)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < iterations)
	{
		int start = i * SumBlockSize;
		int end = min((i + 1) * SumBlockSize, size);
		res_arr[i] = 0;
		for (int j = start; j < end; j++)
			res_arr[i] = __fadd_rn(res_arr[i], arr[j]);
	}
}

int main()
{
	float *arr1, *arr2;
	arr1 = (float*)malloc(sizeof(float) * ArrAmoun);
	arr2 = (float*)malloc(sizeof(float) * ArrAmoun);
	for (int i = 0; i < ArrAmoun; i++)
	{ 
		arr1[i] = rand() / (float)RAND_MAX;
		arr2[i] = rand() / (float)RAND_MAX;
	}
	float *arr1_device, *arr2_device;
	cudaMalloc((void**)&arr1_device, sizeof(float) * ArrAmoun);
	cudaMalloc((void**)&arr2_device, sizeof(float) * ArrAmoun);

	cudaMemcpy(arr1_device, arr1, sizeof(float) * ArrAmoun, cudaMemcpyHostToDevice);
	cudaMemcpy(arr2_device, arr2, sizeof(float) * ArrAmoun, cudaMemcpyHostToDevice);

	int threadsPerBlock = BlockSize;
	int blocksPerGrid = ArrAmoun / threadsPerBlock + (ArrAmoun % threadsPerBlock != 0 ? 1 : 0);
	multiply << <blocksPerGrid , threadsPerBlock>> > (arr1_device, arr2_device);
	cudaDeviceSynchronize();

	bool isOdd = true;
	int arr_length = ArrAmoun;
	int iterations = arr_length / SumBlockSize + (arr_length % SumBlockSize != 0 ? 1 : 0);
	while (arr_length != 1)
	{
		blocksPerGrid = iterations / threadsPerBlock + (iterations % threadsPerBlock != 0 ? 1 : 0);
		if (isOdd)
			sum << <blocksPerGrid, threadsPerBlock >> > (arr1_device, arr2_device, arr_length, iterations);
		else
			sum << <blocksPerGrid, threadsPerBlock >> > (arr2_device, arr1_device, arr_length, iterations);
		cudaDeviceSynchronize();
		arr_length = iterations;
		iterations = arr_length / SumBlockSize + (arr_length % SumBlockSize != 0 ? 1 : 0);
		isOdd = !isOdd;
	}
	float *result;
	result = (float*)malloc(sizeof(float));
	cudaMemcpy(result, isOdd ? arr1_device : arr2_device, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Dot product equals to %0.7f", result[0]);
	cudaFree(arr1_device);
	cudaFree(arr2_device);
	free(arr1);
	free(arr2);
	return 0;
}