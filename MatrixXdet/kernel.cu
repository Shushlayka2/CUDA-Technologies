#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <stdio.h>
#include <stdexcept>
#include <fstream>

using namespace std;

#define N 2
#define Alpha 1.0f
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
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

int main()
{
	float* X = NULL;
	float alpha = 1.0f;
	float beta = 1.0f;
	cublasHandle_t handle;
	float A[N * N] = { 1, 0, 1, 1 }; 
	float B[N * N] = { 12, 6, 10, 8 };
	float C[N * N] = { 1, 0, -1, 1 };
	float I[N * N] = { 1, 0, 0, 1 };
	float* dev_A = NULL, * dev_B = NULL, * dev_C = NULL, * dev_I = NULL, * dev_C_cpy = NULL;
	
	try
	{
		X = (float*)malloc(N * N * sizeof(float));

		cudaMalloc((void**)&dev_A, N * N * sizeof(float));
		cudaMalloc((void**)&dev_B, N * N * sizeof(float));
		cudaMalloc((void**)&dev_C, N * N * sizeof(float));
		cudaMalloc((void**)&dev_I, N * N * sizeof(float));
		cudaMalloc((void**)&dev_C_cpy, N * N * sizeof(float));

		cublascall(cublasCreate(&handle));
		cublascall(cublasSetMatrix(N, N, sizeof(float), A, N, dev_A, N));
		cublascall(cublasSetMatrix(N, N, sizeof(float), B, N, dev_B, N));
		cublascall(cublasSetMatrix(N, N, sizeof(float), C, N, dev_C, N));
		cublascall(cublasSetMatrix(N, N, sizeof(float), C, N, dev_C_cpy, N));
		cublascall(cublasSetMatrix(N, N, sizeof(float), I, N, dev_I, N));

		cublascall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_A, N, dev_I, N, &beta, dev_C, N));

		alpha = Alpha;
		beta = 0.0f;
		cublascall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_C, N, dev_C_cpy, N, &beta, dev_C, N));

		alpha = 1.0f;
		cublascall(cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, &alpha, dev_C, N, dev_B, N));

		cublascall(cublasGetMatrix(N, N, sizeof(float), dev_B, N, X, N));

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				printf("%3.0f ", X[IDX2C(i, j, N)]);
			printf("\n");
		}
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
	}

	free(X);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaFree(dev_I);
	cudaFree(dev_C_cpy);
	cublasDestroy(handle);
	return 0;
}