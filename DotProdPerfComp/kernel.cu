#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublasXt.h> 
#include <cublas_v2.h>

#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <fstream>

using namespace std;

#define N 200
#define AbsMaxVal 10
#define Repetitions 10
#define BLOCK_SIZE 32
#define throw_line(arg) throw custom_exception(arg, __LINE__)
#define cublascall(call) {cublasStatus_t status = (call); if(CUBLAS_STATUS_SUCCESS != status) throw_line("" + status);}

class custom_exception : public runtime_error {
private:
	ofstream log_file;
public:
	custom_exception(const string& arg, int line) :
		runtime_error(arg), log_file("log.txt", ofstream::app) {
		log_file << line << ": " << arg << endl;
	}
	void destroy()
	{
		log_file.close();
	}
};

void generate_random_arr(float* A);
void print_result(float* A, float time);
void cublas_run(float* h_A, float* h_B, float* h_C);
void cublasXt_run(float* h_A, float* h_B, float* h_C);
void simple_mult_run(float* h_A, float* h_B, float* h_C);

__global__ void matrixMult(float* A, float* B, float* C)
{
	int aBegin = N * blockDim.y * blockIdx.y;
	int aEnd = aBegin + N - 1;
	int aStep = blockDim.x;
	int bBegin = blockDim.x * blockIdx.x;
	int bStep = blockDim.y * N;
	__shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];
	if (blockDim.x * blockIdx.x + threadIdx.x <= N && blockDim.y * blockIdx.y + threadIdx.y <= N)
	{
		float sum = 0.0;
		for (int ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep)
		{
			as[threadIdx.y][threadIdx.x] = A[ia + N * threadIdx.y + threadIdx.x];
			bs[threadIdx.y][threadIdx.x] = B[ib + N * threadIdx.y + threadIdx.x];
			__syncthreads();

			for (int k = 0; k < blockDim.x; k++)
				sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
			__syncthreads();
		}
		int ind = N * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
		C[ind] = sum;
	}
}

int main()
{
	float *h_A = NULL, *h_B = NULL, *h_C = NULL;
	try
	{
		srand((unsigned int)time(NULL));
		h_A = (float*)malloc(N * N * sizeof(float));
		h_B = (float*)malloc(N * N * sizeof(float));
		h_C = (float*)malloc(N * N * sizeof(float));
		generate_random_arr(h_A);
		generate_random_arr(h_B);

		cublas_run(h_A, h_B, h_C);
		cublasXt_run(h_A, h_B, h_C);
		simple_mult_run(h_A, h_B, h_C);
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
	}

	free(h_A);
	free(h_B);
	return 0;
}

void cublas_run(float* h_A, float* h_B, float* h_C)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	float KernelTime;
	cublasHandle_t handle;
	cudaEvent_t start, stop;
	float* dev_A = NULL, * dev_B = NULL, * dev_C = NULL;
	try
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaMalloc((void**)&dev_A, N * N * sizeof(float));
		cudaMalloc((void**)&dev_B, N * N * sizeof(float));
		cudaMalloc((void**)&dev_C, N * N * sizeof(float));
		cublascall(cublasCreate(&handle));
		cublascall(cublasSetMatrix(N, N, sizeof(float), h_A, N, dev_A, N));
		cublascall(cublasSetMatrix(N, N, sizeof(float), h_B, N, dev_B, N));
		
		cudaEventRecord(start, 0);
		for (int i = 0; i < Repetitions; i++)
			cublascall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_A, N, dev_B, N, &beta, dev_C, N));

		cublascall(cublasGetMatrix(N, N, sizeof(float), dev_C, N, h_C, N));

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&KernelTime, start, stop);

		printf("Cublas matrixes' multiplication result:\n");
		print_result(h_C, KernelTime);
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
	}
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cublasDestroy(handle);
}

void cublasXt_run(float* h_A, float* h_B, float* h_C)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	int devices[1] = { 0 };
	float KernelTime;
	cudaEvent_t start, stop;
	cublasXtHandle_t handle = NULL;
	try
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cublascall(cublasXtCreate(&handle));
		cublascall(cublasXtDeviceSelect(handle, 1, devices));
		cublascall(cublasXtSetBlockDim(handle, 64));

		cudaEventRecord(start, 0);
		for (int i = 0; i < Repetitions; i++)
			cublascall(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_B, N, &beta, h_C, N));

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&KernelTime, start, stop);

		printf("CublasXt matrixes' multiplication result:\n");
		print_result(h_C, KernelTime);
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cublascall(cublasXtDestroy(handle));
}

void simple_mult_run(float* h_A, float* h_B, float* h_C)
{
	float KernelTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float* dev_A, * dev_B, * dev_C;
	cudaMalloc((void**)&dev_A, sizeof(float) * N * N);
	cudaMalloc((void**)&dev_B, sizeof(float) * N * N);
	cudaMalloc((void**)&dev_C, sizeof(float) * N * N);
	cudaMemcpy(dev_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1), N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1));

	cudaEventRecord(start, 0);
	for (int i = 0; i < Repetitions; i++)
		matrixMult << <blocksPerGrid, threadsPerBlock >> > (dev_A, dev_B, dev_C);

	cudaMemcpy(h_C, dev_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&KernelTime, start, stop);

	printf("Simple matrixes' multiplication result:\n");
	print_result(h_C, KernelTime);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void generate_random_arr(float* A)
{
	for (int i = 0; i < N * N; i++)
	{
		A[i] = i; // (rand() / (float)RAND_MAX)* (AbsMaxVal * 2 + 1) + -1 * AbsMaxVal;
	}
}

void print_result(float* A, float time)
{
	printf("Elapsed time equals to: %f\n", time / Repetitions);
	printf("Result matrix equals to:\n");

	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%f ", A[i * N + j]);
		}
		printf("\n");
	}*/
	printf("\n");
}