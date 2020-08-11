#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <stdio.h>
#include <vector>

using namespace std;

#define BLOCK_SIZE 256

__global__ void self_dot_prod(const float* A, float* B, int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length)
	{
		float sum = 0.0f;
		__shared__ float ash[BLOCK_SIZE];
		ash[threadIdx.x] = A[i];
		__syncthreads();
		if (threadIdx.x == 0)
		{
			sum = 0.0;
			for (int j = 0; j < blockDim.x; j++)
				sum += ash[j] * ash[j];

			B[blockIdx.x] = sum;
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
	float* A, * B;
	A = &data[0];
	B = (float*)malloc(sizeof(float) * length);
	float* d_A, * d_B;
	cudaMalloc((void**)&d_A, sizeof(float) * length);
	cudaMalloc((void**)&d_B, sizeof(float) * length);
	cudaMemcpy(d_A, A, sizeof(float) * length, cudaMemcpyHostToDevice);
	
	int GRID_SIZE = length / BLOCK_SIZE + (length % BLOCK_SIZE != 0 ? 1 : 0);
	self_dot_prod << <GRID_SIZE, BLOCK_SIZE >> > (d_A, d_B, length);
	cudaMemcpy(B, d_B, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
		result += B[i];
	printf("Result: %f\n", sqrt(result));

	free(B);
	cudaFree(d_A);
	cudaFree(d_B);
	data.clear();
}