#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define BLOCK_SIZE 10

__global__ void calc(float* A, float* result, int size)
{
	int indexY = size * (blockDim.y * blockIdx.y + threadIdx.y);
	int indexX = size * (blockDim.x * blockIdx.x + threadIdx.x);
	if (blockDim.y * blockIdx.y + threadIdx.y <= size && blockDim.x * blockIdx.x + threadIdx.x <= size)
	{
		float sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += A[indexY + i] * A[indexX + i];
		result[size * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x] = sum;
	}
}

bool compare_with_identity(float* A, int size);

int main()
{
	ifstream file;
	file.open("Input.txt");
	vector<float> data;
	int length = 0;
	while (!file.eof())
	{
		float a;
		file >> a;
		data.push_back(a);
		length++;
	}
	file.close();

	float* d = &data[0];
	float* result;
	float* d_device;
	float* result_device;
	result = (float*)malloc(sizeof(float) * length);
	cudaMalloc((void**)&d_device, sizeof(float) * length);
	cudaMalloc((void**)&result_device, sizeof(float) * length);
	cudaMemcpy(d_device, d, sizeof(float) * length, cudaMemcpyHostToDevice);
	int size = sqrt(length);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE); 
	dim3 blocksPerGrid = dim3(size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1), size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1));

	calc << <blocksPerGrid, threadsPerBlock >> > (d_device, result_device, size);
	cudaMemcpy(result, result_device, sizeof(float) * length, cudaMemcpyDeviceToHost);

	if (compare_with_identity(result, size))
		printf("Matrix is Orthogonal");
	else
		printf("Matrix isn't Orthogonal");

	printf("\n");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%f ", result[i * size + j]);
		}
		printf("\n");
	}

	data.clear();
	free(result);
	cudaFree(d_device);
	cudaFree(result_device);
	return 0;
}

bool compare_with_identity(float* A, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (i == j)
			{
				if (A[i * size + j] != 1)
					return false;
			}
			else
			{
				if (A[i * size + j] != 0)
					return false;
			}
		}
	}
	return true;
}