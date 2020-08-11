#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

#define BLOCK_SIZE 10

__global__ void check(float* A, float* B, int* result, int size)
{
	int indexY = size * (blockDim.y * blockIdx.y + threadIdx.y);
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	if (blockDim.y * blockIdx.y + threadIdx.y <= size && blockDim.x * blockIdx.x + threadIdx.x <= size)
	{
		float sumAB = 0.0f;
		float sumBA = 0.0f;
		for (int i = 0; i < size; i++)
		{
			sumAB += A[indexY + i] * B[i * size + indexX];
			sumBA += B[indexY + i] * A[i * size + indexX];
		}
		if (sumAB != sumBA)
			*result = atomicOr(result, 1);
	}
}

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

	length /= 2;
	int size = sqrt(length);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1), size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1));

	float* h_A = &data[0];
	float* h_B = &data[length];	
	float* d_A;
	float* d_B;
	int* h_result = new int;
	int* d_result;
	cudaMalloc((void**)&d_A, sizeof(float) * length);
	cudaMalloc((void**)&d_B, sizeof(float) * length);
	cudaMemcpy(d_A, h_A, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_result, sizeof(int));

	check << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_result, size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

	if (*h_result == 1)
		printf("Matrixes are non commuting");
	else
		printf("Matrixes are commuting");

	delete h_result;
	cudaFree(d_result);
	cudaFree(d_A);
	cudaFree(d_B);
	data.clear();
	return 0;
}