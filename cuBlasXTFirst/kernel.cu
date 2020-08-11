#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublasXt.h> 

#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <fstream>

using namespace std;

#define N 2
#define op CUBLAS_OP_N
#define AbsMaxVal 10
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

void geberate_identity_matrix(float* A);
void generate_random_arr(float* A);

int main()
{
	float* h_A = NULL, * h_B = NULL, * h_C = NULL, * h_I = NULL;
	cublasXtHandle_t handle = NULL;
	try
	{
		h_I = (float*)calloc(N * N, sizeof(h_I[0]));
		h_A = (float*)malloc(N * N * sizeof(h_A[0]));
		h_B = (float*)malloc(N * N * sizeof(h_B[0]));
		h_C = (float*)malloc(N * N * sizeof(h_C[0]));

		srand((unsigned int)time(NULL));
		geberate_identity_matrix(h_I);
		generate_random_arr(h_A);
		generate_random_arr(h_B);

		int devices[1] = { 0 };
		cublascall(cublasXtCreate(&handle));
		cublascall(cublasXtDeviceSelect(handle, 1, devices));
		cublascall(cublasXtSetBlockDim(handle, 64));

		float alpha = 1.0f;
		float beta = 0.0f;
		cublascall(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_B, N, &beta, h_C, N));

		beta = 1.0f;
		cublascall(cublasXtSgemm(handle, op, CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_I, N, &beta, h_B, N));

		beta = 0.0f;
		cublascall(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, h_B, N, h_C, N, &beta, h_B, N));

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("%f ", h_B[i * N + j]);
			}
			printf("\n");
		}
	}
	catch (custom_exception &ex)
	{
		ex.destroy();
	}
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_I);
	cublasXtDestroy(handle);
	return 0;
}

void geberate_identity_matrix(float* A)
{
	for (int i = 0; i < N; i++)
	{
		A[i * N + i] = 1;
	}
}

void generate_random_arr(float* A)
{
	for (int i = 0; i < N * N; i++)
	{
		A[i] = (rand() / (float)RAND_MAX) * (AbsMaxVal * 2 + 1) + -1 * AbsMaxVal; //i;
	}
}