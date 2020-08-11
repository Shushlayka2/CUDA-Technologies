#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
using namespace std;

#define N 10
#define BlockSize 32
#define SumBlockSize 16

float sumElems(float* Temp, float* Temp2);
float dotProduct(float* A, float* B, float* Temp, float* Temp2, int leftA, int rightA, int leftB, int rightB);

__global__ void multiply(float* A, float* B, float* Result, int leftA, int rightA, int leftB, int rightB)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = index + leftA;
	int j = index + leftB;
	if (i < rightA)
	{
		Result[index] = A[i] * B[j];
	}
}

__global__ void multiply_num_vect(float* A, float* Result, float num, int left, int right)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = index + left;
	if (i < right)
	{
		Result[index] = A[i] * num;
		//printf("%f\n", Result[index]);
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
		{
			res_arr[i] = res_arr[i] + arr[j];
		}		
	}
}

__global__ void sum_elems(float* arr, float* res_arr, int leftA, int rightA, int leftB, int rightB)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = index + leftA;
	int j = index + leftB;
	if (i < rightA)
	{
		//printf("%f + %f = %f\n", arr[i], res_arr[j], res_arr[j] + arr[i]);
		res_arr[j] = res_arr[j] + arr[i];
	}
}

int main()
{
	size_t size = N * N;
	float* h_A, *h_B;
	h_A = (float*)calloc(size, sizeof(float));
	h_B = (float*)calloc(size, sizeof(float));
	for (int i = 0; i < N; i++)
	{
		for (int j = (N - 1) - i; j >= 0; j--)
		{
			h_A[i * N + j]= 1;
		}
		/*for (int j = 0; j < N; j++)
			printf("%i", arr[i * N + j]);
		printf("\n");*/
	}
	for (int i = 0; i < N; i++)
	{
		h_B[i] = 1;
	}
	
	float* d_A, * d_B, *d_Temp, *d_Temp2;
	cudaMalloc((void**)&d_A, sizeof(float) * size);
	cudaMalloc((void**)&d_B, sizeof(float) * size);
	cudaMalloc((void**)&d_Temp, sizeof(float) * size);
	cudaMalloc((void**)&d_Temp2, sizeof(float) * size);
	cudaMemcpy(d_A, h_A, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * size, cudaMemcpyHostToDevice);

	int GridSize = N / BlockSize + (N % BlockSize != 0 ? 1 : 0);
	float* divider;
	divider = (float*)calloc(N, sizeof(float));

	for (int i = 1; i < N; i++)
	{
		divider[i - 1] = dotProduct(d_B, d_B, d_Temp, d_Temp2, (i - 1) * N, i * N, (i - 1) * N, i * N);
		for (int k = 0; k < i; k++)
		{
			float dividend = dotProduct(d_A, d_B, d_Temp, d_Temp2, i * N, (i + 1) * N, k * N, (k + 1) * N);
			//printf("%f %f\n", dividend, divider[k]);
			multiply_num_vect << <GridSize, BlockSize >> > (d_B, d_Temp, -1 * dividend / divider[k], k * N, (k + 1) * N);
			sum_elems << <GridSize, BlockSize >> > (d_Temp, d_B, 0, N, i * N, (i + 1) * N);
		}
		sum_elems << <GridSize, BlockSize >> > (d_A, d_B, i * N, (i + 1) * N, i * N, (i + 1) * N);
	}

	cudaMemcpy(h_B, d_B, sizeof(float) * size, cudaMemcpyDeviceToHost);
	printf("\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			printf("%f ", h_B[i * N + j]);
		}
		printf("\n");
	}

	free(h_A);
	free(h_B);
	free(divider);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_Temp);
	cudaFree(d_Temp2);

	return 0;
}

float dotProduct(float* A, float* B, float* Temp, float* Temp2, int leftA, int rightA, int leftB, int rightB)
{
	int GridSize = N / BlockSize + (N % BlockSize != 0 ? 1 : 0);
	multiply << <GridSize, BlockSize >> > (A, B, Temp, leftA, rightA, leftB, rightB);
	cudaDeviceSynchronize();
	return sumElems(Temp, Temp2);
}

float sumElems(float* Temp, float* Temp2)
{
	bool isOdd = true;
	int arr_length = N;
	int iterations = arr_length / SumBlockSize + (arr_length % SumBlockSize != 0 ? 1 : 0);
	while (arr_length != 1)
	{
		int GridSize = iterations / BlockSize + (iterations % BlockSize != 0 ? 1 : 0);
		if (isOdd)
			sum << <GridSize, BlockSize >> > (Temp, Temp2, arr_length, iterations);
		else
			sum << <GridSize, BlockSize >> > (Temp2, Temp, arr_length, iterations);
		cudaDeviceSynchronize();
		arr_length = iterations;
		iterations = arr_length / SumBlockSize + (arr_length % SumBlockSize != 0 ? 1 : 0);
		isOdd = !isOdd;
	}
	float result;
	cudaMemcpy(&result, isOdd ? Temp : Temp2, sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}