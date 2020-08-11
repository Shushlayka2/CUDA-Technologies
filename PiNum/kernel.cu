#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <math.h>

using namespace std;

__global__ void calc(float* arr, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	float val = arr[x];
	if (x < n) {
		arr[x] = sqrtf(1 - val * val);
	}
}

int main()
{
	float* arr;
	float* arr_device;
	int parts_amount = 10000;
	arr = (float*)malloc(sizeof(float) * parts_amount);
	cudaMalloc((void**)&arr_device, sizeof(float) * parts_amount);
	int blockSize = 5;
	int gridSize = parts_amount / blockSize + (parts_amount % blockSize == 0 ? 0 : 1);
	float left_edge = 0.0f, right_edge = 1.0f;
	float step = (right_edge - left_edge) / parts_amount;
	int i = 0;
	for (float x = left_edge + step; x <= right_edge; x += step)
	{
		arr[i] = x;
		i++;
	}
	cudaMemcpy(arr_device, arr, sizeof(float) * parts_amount, cudaMemcpyHostToDevice);
	calc << < gridSize, blockSize >> > (arr_device, parts_amount);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("%s", cudaGetErrorString(err));
	}
	else
	{
		cudaMemcpy(arr, arr_device, sizeof(float) * parts_amount, cudaMemcpyDeviceToHost);
		float sum = 0.0f;
		for (int j = 0; j < i; j++) {
			sum += arr[j];
		}
		printf("Pi equals to %0.7f\n", sum / parts_amount * 4.0f);
	}
	return 0;
}