#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <malloc.h>
#include <stdio.h>
#include <stdexcept>
#include <fstream>

using namespace std;

#define N 2
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
	cublasHandle_t handle;
	float* A, *b, *x;
	float** dev_A, **hostPointer_devA;
	float** dev_b, **hostPointer_devb;
	int* d_pivot_array;
	int* d_info_array;

	try
	{
		x = (float*)malloc(N * sizeof(float));
		b = (float*)malloc(N * sizeof(float));
		A = (float*)malloc(N * N * sizeof(float));
		for (int j = 0; j < N; j++) //N cols
		{
			for (int i = 0; i < N; i++) //M rows
				A[IDX2C(i, j, N)] = (float)(i * N + j + 1);
			b[IDX2C(0, j, 1)] = 1.0f;
		}

		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < N; i++)
				printf("%3.0f ", A[IDX2C(i, j, N)]);
			printf("\n");
		}
		printf("\n");

		cudaMalloc((void**)&d_pivot_array, N * sizeof(int));
		cudaMalloc((void**)&d_info_array, sizeof(int));

		hostPointer_devb = (float**)malloc(sizeof(float*));
		cudaMalloc((void**)&hostPointer_devb[0], N * sizeof(float));
		cudaMalloc((void**)&dev_b, sizeof(float*));
		
		hostPointer_devA = (float**)malloc(sizeof(float*));
		cudaMalloc((void**)&hostPointer_devA[0], N * N * sizeof(float));
		cudaMalloc((void**)&dev_A, sizeof(float*));
		cudaMemcpy(dev_b, hostPointer_devb, sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_A, hostPointer_devA, sizeof(float*), cudaMemcpyHostToDevice);

		cublasCreate(&handle);
		cublascall(cublasSetMatrix(N, N, sizeof(float), A, N, hostPointer_devA[0], N));
		cublascall(cublasSetMatrix(N, 1, sizeof(float), b, N, hostPointer_devb[0], N));

		cublascall(cublasGetMatrix(N, N, sizeof(float), hostPointer_devA[0], N, A, N));
		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < N; i++)
				printf("%3.0f ", A[IDX2C(i, j, N)]);
			printf("\n");
		}
		printf("\n");

		cublascall(cublasSgetrfBatched(handle, N, dev_A, N, d_pivot_array, d_info_array, 1));

		cublascall(cublasGetMatrix(N, N, sizeof(float), hostPointer_devA[0], N, A, N));
		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < N; i++)
				printf("%3.0f ", A[IDX2C(i, j, N)]);
			printf("\n");
		}
		printf("\n");

		cublascall(cublasSgetrsBatched(handle, CUBLAS_OP_N, N, 1, dev_A, N, d_pivot_array, dev_b, N, d_info_array, 1));
		cublascall(cublasGetMatrix(N, 1, sizeof(float), hostPointer_devb[0], N, x, N));
		
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				printf("%3.0f ", A[IDX2C(i, j, N)]);
			printf(" = %f %4.6f\n", b[i], x[i]);
		}
	}
	catch (custom_exception &ex)
	{
		ex.destroy();
	}

	free(x);
	free(b);
	free(A);
	cudaFree(hostPointer_devA[0]);
	cudaFree(hostPointer_devb[0]);
	free(hostPointer_devA);
	free(hostPointer_devb);
	cudaFree(dev_A);
	cudaFree(dev_b);
	cublasDestroy(handle);
	return 0;
}