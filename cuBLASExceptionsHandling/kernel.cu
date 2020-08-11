#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <malloc.h>
#include <stdio.h>
#include <stdexcept>
#include <fstream>

using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define throw_line(arg) throw custom_exception(arg, __LINE__)
#define cublascall(call) {cublasStatus_t status = (call); if(CUBLAS_STATUS_SUCCESS != status) throw_line("" + status);}
#define cudacall(call) {cudaError_t err = (call); if(cudaSuccess != err) throw_line(cudaGetErrorString(err));}

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
	int prog_status = 0;
	const int N = 6;
	cublasHandle_t handle = NULL;
	float* dev_A = NULL, * dev_b = NULL;
	float* x = NULL, * A = NULL, * b = NULL;
	try
	{
		x = (float*)malloc(N * sizeof(*x));
		b = (float*)malloc(N * sizeof(*b));
		A = (float*)malloc(N * N * sizeof(*A));
		int ind = 11;
		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < N; i++)
				if (i >= j)
					A[IDX2C(i, j, N)] = (float)ind++;
				else A[IDX2C(i, j, N)] = 0.0f;
			b[j] = 1.0f;
		}

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("% 3.0f", A[i * N + j]);
			}
			printf("\n");
		}
		cudacall(cudaMalloc((void**)&dev_b, N * sizeof(*x)));
		cudacall(cudaMalloc((void**)&dev_A, N * N * sizeof(*A)));
		
		cublascall(cublasCreate(&handle));
		cublascall(cublasSetVector(N, sizeof(*b), b, 1, dev_b, 1));
		cublascall(cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N));

		cublascall(cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, dev_A, N, dev_b, 1));

		cublascall(cublasGetVector(N, sizeof(*x), dev_b, 1, x, 1));

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				printf("%3.0f ", A[IDX2C(i, j, N)]);
			printf(" = %f %4.6f\n", b[i], x[i]);
		}
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
		prog_status = 1;
	}
	cudaFree(dev_b);
	cudaFree(dev_A);
	cublasDestroy(handle);
	free(x);
	free(b);
	free(A);
	return prog_status;
}