#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>

using namespace std;

#define f __fsqrt_rn
#define a 0
#define b 1
#define N 1000
#define BLOCK_SIZE 128

float sum_particles_host(float* d_A_even, float* d_A_odd);

texture<float, 1, cudaReadModeElementType> StepRef;
texture<float, 1, cudaReadModeElementType> ParticlesRef;

__global__ void calc_rectangle(float* A)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < N)
        A[idx] = f(a + tex1Dfetch(StepRef, 0) * (idx + 0.5f));
}

__global__ void sum_particles(float* A, int size, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iterations)
    {
        int start = idx * BLOCK_SIZE;
        int end = min((idx + 1) * BLOCK_SIZE, size);
        A[idx] = 0;
        for (int j = start; j < end; j++)
            A[idx] = __fadd_rn(A[idx], tex1Dfetch(ParticlesRef, j));
    }
}

int main()
{
    int GRID_SIZE = N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1);
    float step = (b - a) / (float)N;
    float* d_step, *d_A_even, *d_A_odd;
    cudaMalloc((void**)&d_step, sizeof(float));
    cudaMalloc((void**)&d_A_even, sizeof(float) * N);
    cudaMalloc((void**)&d_A_odd, sizeof(float) * N);
    cudaMemcpy(d_step, &step, sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(0, StepRef, d_step, sizeof(float));

    calc_rectangle << <GRID_SIZE, BLOCK_SIZE >> > (d_A_odd);
    cudaDeviceSynchronize();

    cudaBindTexture(0, ParticlesRef, d_A_odd, sizeof(float) * N);

    float result = sum_particles_host(d_A_even, d_A_odd);
    printf("Result: %f", result * step);

    cudaUnbindTexture(StepRef);
    cudaFree(d_step);
    cudaFree(d_A_even);
    cudaFree(d_A_odd);
	return 0;
}

float sum_particles_host(float* d_A_even, float* d_A_odd)
{
    bool isOdd = true;
    int arr_length = N;
    int iterations = arr_length / BLOCK_SIZE + (arr_length % BLOCK_SIZE != 0 ? 1 : 0);
    while (arr_length != 1)
    {
        int GRID_SIZE = iterations / BLOCK_SIZE + (iterations % BLOCK_SIZE != 0 ? 1 : 0);
        if (isOdd)
        {
            sum_particles << <GRID_SIZE, BLOCK_SIZE >> > (d_A_even, arr_length, iterations);
            cudaBindTexture(0, ParticlesRef, d_A_even, sizeof(float) * iterations);
        }
        else
        {
            sum_particles << <GRID_SIZE, BLOCK_SIZE >> > (d_A_odd, arr_length, iterations);
            cudaBindTexture(0, ParticlesRef, d_A_odd, sizeof(float) * iterations);
        }
        cudaDeviceSynchronize();
        
        arr_length = iterations;
        iterations = arr_length / BLOCK_SIZE + (arr_length % BLOCK_SIZE != 0 ? 1 : 0);
        isOdd = !isOdd;
    }
    float* result;
    result = (float*)malloc(sizeof(float));
    cudaMemcpy(result, isOdd ? d_A_odd : d_A_even, sizeof(float), cudaMemcpyDeviceToHost);
    return *result;
}