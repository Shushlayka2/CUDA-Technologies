#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <ctime>

#define M_PI 3.14159265358979323846
#define N 409600
#define R 102400
#define SquaredR 10485760000

__global__ void optimized_calc(unsigned int* count, unsigned int seed, unsigned int iterations_count)
{
	curandState_t state;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, i, 0, &state);
	unsigned int local_count = 0;
	for (int j = 0; j < iterations_count; j++)
	{
		float x = curand_uniform(&state) * R;
		float y = curand_uniform(&state) * R;
		if (y * y <= SquaredR - x * x)
		{
			local_count++;
		}
	}
	atomicAdd(count, local_count);
}

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
	unsigned int* count_device;
	cudaMalloc((void**)&count_device, sizeof(unsigned int));

	unsigned int blockSize = 32;
	unsigned int iterations_count = 100;
	unsigned int gridSize = N / (iterations_count * blockSize) + (N % (iterations_count * blockSize) == 0 ? 0 : 1);
	clock_t begin = clock();
	optimized_calc << <gridSize, blockSize >> > (count_device, time(NULL), iterations_count);
	cudaDeviceSynchronize();
	clock_t end = clock();
	auto elapsedOptimizedTime = double(end - begin) / CLOCKS_PER_SEC;
	cudaMemcpy(&count, count_device, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	float PiOptimized = (count / (float)N) * 4.0f;
	float optimizedError = fabs(PiOptimized - M_PI);

	unsigned int* count_device_optimized;
	cudaMalloc((void**)&count_device_optimized, sizeof(unsigned int));
	gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);
	begin = clock();
	calc << <gridSize, blockSize>> > (count_device_optimized, time(NULL));
	cudaDeviceSynchronize();
	end = clock(); 
	auto elapsedTime = double(end - begin) / CLOCKS_PER_SEC;
	cudaMemcpy(&count, count_device_optimized, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	float Pi = (count / (float)N) * 4.0f;
	float error = fabs(Pi - M_PI);

	cudaFree(count_device);
	cudaFree(count_device_optimized);

	printf("Ordinary:\n");
	printf("Error: %f\n", error);
	printf("Elapsed Time: %lf sec\n", elapsedTime);
	printf("Pi equals to %f\n\n", Pi);
	printf("Optimized:\n");
	printf("Error: %f\n", optimizedError);
	printf("Elapsed Time: %lf sec\n", elapsedOptimizedTime);
	printf("Pi equals to %f", PiOptimized);
	return 0;
}