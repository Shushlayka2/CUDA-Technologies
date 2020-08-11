#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define BLOCK_SIZE 10

__global__ void matrixAdd(float** A, float** B, float** C, int M, int N) 
{ 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y < M && x < N)
	{
		C[y][x] = A[y][x] + B[y][x];
		//printf("%f + %f = %f\n", A[y][x], B[y][x], C[y][x]);
	}
}

int main()
{
	ifstream file;
	int M, N;
	file.open("Input.txt");
	file >> M >> N;
	float **A = new float*[M];
	float **B = new float*[M];
	float **C = new float*[M];

	for (int i = 0; i < M; i++)
	{
		A[i] = new float[N];
		C[i] = new float[N];
		for (int j = 0; j < N; j++)
		{
			file >> A[i][j];
		}
	}
	for (int i = 0; i < M; i++)
	{
		B[i] = new float[N];
		for (int j = 0; j < N; j++)
		{
			file >> B[i][j];
		}
	}
	file.close();

	float **d_A, **d_B, **d_C;
	cudaMalloc((void**)&d_A, sizeof(float*) * M);
	cudaMalloc((void**)&d_B, sizeof(float*) * M);
	cudaMalloc((void**)&d_C, sizeof(float*) * M);
	float **s_A, **s_B, **s_C;
	s_A = (float**)malloc(sizeof(float*) * M);
	s_B = (float**)malloc(sizeof(float*) * M);
	s_C = (float**)malloc(sizeof(float*) * M);
	for (int i = 0; i < M; i++)
	{
		cudaMalloc((void**)&s_A[i], sizeof(float) * N);
		cudaMalloc((void**)&s_B[i], sizeof(float) * N);
		cudaMalloc((void**)&s_C[i], sizeof(float) * N);
		cudaMemcpy(s_A[i], A[i], sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(s_B[i], B[i], sizeof(float) * N, cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_A, s_A, sizeof(float*) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, s_B, sizeof(float*) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, s_C, sizeof(float*) * M, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(M / BLOCK_SIZE + (M % BLOCK_SIZE == 0 ? 0 : 1), N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1));
	matrixAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, M, N);

	for (int i = 0; i < M; i++)
	{
		cudaMemcpy(C[i], s_C[i], sizeof(float) * N, cudaMemcpyDeviceToHost);
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%f ", C[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < M; i++)
	{
		delete[] A[i];
		delete[] B[i];
		delete[] C[i];
	}
	delete[] A;
	delete[] B;
	delete[] C;
	for (int i = 0; i < M; i++)
	{
		cudaFree(s_A[i]);
		cudaFree(s_B[i]);
		cudaFree(s_C[i]);
	}
	free(s_A);
	free(s_B);
	free(s_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}