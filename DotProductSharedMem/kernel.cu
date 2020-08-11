#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define BLOCK_SIZE 256

__global__ void scalMult(const float* A, const float* B, float* C, int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length)
	{
		float sum = 0.0f;
		__shared__ float ash[BLOCK_SIZE];
		__shared__ float bsh[BLOCK_SIZE];
		ash[threadIdx.x] = A[i];
		bsh[threadIdx.x] = B[i];
		__syncthreads();
		if (threadIdx.x == 0)
		{
			sum = 0.0;
			for (int j = 0; j < blockDim.x; j++)
				sum += ash[j] * bsh[j];

			C[blockIdx.x] = sum;
		}
	}
}
int main()
{
	ifstream file;
	file.open("input.txt");
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
	float *A, * B, *C;
	A = &data[0];
	B = &data[length];
	C = (float*)malloc(sizeof(float) * length);
	
	float *d_A, * d_B, *d_C;
	cudaMalloc((void**)&d_A, sizeof(float) * length);
	cudaMalloc((void**)&d_B, sizeof(float) * length);
	cudaMalloc((void**)&d_C, sizeof(float) * length);
	cudaMemcpy(d_A, A, sizeof(float) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(float) * length, cudaMemcpyHostToDevice);

	float KernelTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int GRID_SIZE = length / BLOCK_SIZE + (length % BLOCK_SIZE != 0 ? 1 : 0);
	cudaEventRecord(start, 0);
	scalMult << <GRID_SIZE, BLOCK_SIZE>> > (d_A, d_B, d_C, length);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&KernelTime, start, stop);

	cudaMemcpy(C, d_C, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
		result += C[i];
	printf("Result: %f\nElapsedTime: %f", result, KernelTime);

	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	data.clear();
	return 0;
}