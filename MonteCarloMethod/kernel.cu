#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

using namespace std;

#define N 409600
#define R 102400
#define SquaredR 10485760000

__global__ void calc(unsigned int* count, unsigned int seed)
{
	curandState_t state;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, i, 0, &state);
	float x = curand_uniform(&state) * R;
	float y = curand_uniform(&state) * R;
	if (y * y <= SquaredR - x * x)
	{
		atomicInc(count, N);
	}
}

int main()
{
	unsigned int count;
	unsigned int *count_device;
	cudaMalloc((void**)&count_device, sizeof(unsigned int));

	int blockSize = 32;
	int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	calc << <gridSize, blockSize >> > (count_device, time(NULL));
	cudaDeviceSynchronize();
	cudaMemcpy(&count, count_device, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printf("%f", (count / (float)N) * 4.0f);
	cudaFree(count_device);
	return 0;
}