#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

#define f sqrt
#define f_core __fsqrt_rn
#define a 0
#define b 1
#define N 1000
#define HalfN N / 2
#define RealRes 0.666667

#define BLOCK_SIZE 128

float calc_rectangle_host(float* d_A, float* A, int GRID_SIZE, float h);
float calc_trapezoid_host(float* d_A, float* A, int GRID_SIZE, float h);
float calc_simpson_host(float* d_A, float* A, int GRID_SIZE, float h);

__global__ void calc_rectangle(float *A, float h)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x;
	__shared__ float sm[BLOCK_SIZE * 2];
	if (i < N)
	{
		sm[threadIdx.x] = f_core(a + h * (i + 0.5f));
		__syncthreads();
		if (threadIdx.x == 0)
		{
			float sum = 0.0f;
			for (int j = 0; j < blockDim.x; j++)
				sum += sm[j];
			A[blockIdx.x] = sum;
		}
	}
}

__global__ void calc_trapezoid(float* A, float h)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float sm[BLOCK_SIZE];
	if (i < N)
	{
		sm[threadIdx.x] = (f_core(a + i * h) + f_core(a + (i + 1) * h)) / 2;
		__syncthreads();
		if (threadIdx.x == 0)
		{
			float sum = 0.0f;
			for (int j = 0; j < blockDim.x; j++)
				sum += sm[j];
			A[blockIdx.x] = sum;
		}
	}
}

__global__ void calc_simpson(float* A, float h)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	__shared__ float sm_even[BLOCK_SIZE];
	__shared__ float sm_odd[BLOCK_SIZE];
	sm_even[threadIdx.x] = 0;
	sm_odd[threadIdx.x] = 0;
	if (i < HalfN)
	{
		sm_even[threadIdx.x] = f_core(a + 2 * i * h);
		sm_odd[threadIdx.x] = f_core(a + (2 * i - 1) * h);
	}
	if (i == HalfN)
	{
		sm_odd[threadIdx.x] = f_core(a + (2 * i - 1) * h);
	}
	__syncthreads();
	if (threadIdx.x == 0)
	{
		float sum_even = 0.0f;
		float sum_odd = 0.0f;
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			sum_even += sm_even[j];
			sum_odd += sm_odd[j];
		}
		A[blockIdx.x] = 2 * sum_even + 4 * sum_odd;
	}
}

int main()
{
	int GRID_SIZE = N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1);
	
	float *A;
	A = (float*)malloc(sizeof(float) * GRID_SIZE);
	float h = (b - a) / (float)N;

	float* d_A;
	cudaMalloc((void**)&d_A, sizeof(float) * GRID_SIZE);

	float result = calc_rectangle_host(d_A, A, GRID_SIZE, h) * h;
	printf("RectangleRule result: %f\n", result);
	printf("Residual: %f\n", RealRes - result);

	result = calc_trapezoid_host(d_A, A, GRID_SIZE, h) * h;
	printf("TrapezoidRule result: %f\n", result);
	printf("Residual: %f\n", RealRes - result);

	result = calc_simpson_host(d_A, A, GRID_SIZE, h) * (h / 3);
	printf("SimpsonRule result: %f\n", result);
	printf("Residual: %f\n", RealRes - result);

	free(A);
	cudaFree(d_A);
	return 0;
}

float calc_rectangle_host(float* d_A, float* A, int GRID_SIZE, float h)
{
	calc_rectangle << <GRID_SIZE, BLOCK_SIZE >> > (d_A, h);
	cudaMemcpy(A, d_A, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
	{
		result += A[i];
	}
	return result;
}

float calc_trapezoid_host(float* d_A, float* A, int GRID_SIZE, float h)
{
	calc_trapezoid << <GRID_SIZE, BLOCK_SIZE >> > (d_A, h);
	cudaMemcpy(A, d_A, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
	{
		result += A[i];
	}
	return result;
}

float calc_simpson_host(float* d_A, float* A, int GRID_SIZE, float h)
{
	GRID_SIZE = HalfN / BLOCK_SIZE + (HalfN % BLOCK_SIZE == 0 ? 0 : 1);
	calc_simpson << <GRID_SIZE, BLOCK_SIZE >> > (d_A, h);
	cudaMemcpy(A, d_A, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);

	float result = 0.0f;
	for (int i = 0; i < GRID_SIZE; i++)
	{
		result += A[i];
	}
	float firstVal = f(a);
	float lastVal = f(b);
	result = firstVal + result + lastVal;
	return result;
}