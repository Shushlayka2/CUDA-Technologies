#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <malloc.h>
#include <vector>

using namespace std;

#define BLOCK_SIZE 128
#define ArraySize 100000
#define AbsMaxVal 10

void generate_random_arr(float* A);
void exec_first_condition(float* A, float* B);
void exec_second_condition(float* A, float* B);
void exec_third_condition(float* A, float* B);
float sum_particles_host(float* d_A_even, float* d_A_odd);

texture<float, 1, cudaReadModeElementType> FirstArrElementsRef;
texture<float, 1, cudaReadModeElementType> SecondArrElementsRef;
texture<float, 1, cudaReadModeElementType> SumArrElementsRef;

__global__ void mult_particles_first(float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ArraySize)
        C[i] = __fmul_rn(tex1D(FirstArrElementsRef, i), tex1D(SecondArrElementsRef, i));
}

__global__ void mult_particles_second(float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ArraySize)
        C[i] = __fmul_rn(tex1Dfetch(FirstArrElementsRef, i), tex1D(SecondArrElementsRef, i));
}

__global__ void mult_particles_third(float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ArraySize)
        C[i] = __fmul_rn(tex1Dfetch(FirstArrElementsRef, i), tex1Dfetch(SecondArrElementsRef, i));
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
            A[idx] = __fadd_rn(A[idx], tex1Dfetch(SumArrElementsRef, j));
    }
}

int main()
{
    /*ifstream file;
    file.open("input.txt");
    vector<float> data;
    int length = 0;
    while (!file.eof())
    {
        float a;
        file >> a;
        data.push_back(a);
        length++;
    }
    file.close();
    length /= 2;
    float* A, * B;
    A = &data[0];
    B = &data[length];*/

    float* A, * B;
    A = (float*)malloc(sizeof(float) * ArraySize);
    B = (float*)malloc(sizeof(float) * ArraySize);
    generate_random_arr(A);
    generate_random_arr(B);

    exec_first_condition(A, B);
    exec_second_condition(A, B);
    exec_third_condition(A, B);
    
    //data.clear();
    cudaUnbindTexture(FirstArrElementsRef);
    cudaUnbindTexture(SecondArrElementsRef);
    cudaUnbindTexture(SumArrElementsRef);
	return 0;
}

void exec_first_condition(float* A, float* B)
{
    float* d_C_odd, *d_C_even;
    cudaArray* d_A, * d_B;
    size_t size = sizeof(float) * ArraySize;
    int GRID_SIZE = ArraySize / BLOCK_SIZE + (ArraySize % BLOCK_SIZE != 0 ? 1 : 0);
    cudaMalloc((void**)&d_C_odd, size);
    cudaMalloc((void**)&d_C_even, size);
    cudaMallocArray(&d_A, &FirstArrElementsRef.channelDesc, ArraySize, 1);
    cudaMallocArray(&d_B, &SecondArrElementsRef.channelDesc, ArraySize, 1);
    cudaMemcpyToArray(d_A, 0, 0, A, size, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(d_B, 0, 0, B, size, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(FirstArrElementsRef, d_A);
    cudaBindTextureToArray(SecondArrElementsRef, d_B);

    float KernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    mult_particles_first << <GRID_SIZE, BLOCK_SIZE >> > (d_C_odd);
    cudaBindTexture(0, SumArrElementsRef, d_C_odd, size);
    float result = sum_particles_host(d_C_even, d_C_odd);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    printf("First condition:\n");
    printf("Result: %f\n", result);
    printf("Elapsed time: %f\n", KernelTime);
    printf("\n");

    cudaUnbindTexture(SumArrElementsRef);
    cudaFree(d_C_odd);
    cudaFree(d_C_even);
    cudaFreeArray(d_A);
    cudaFreeArray(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void exec_second_condition(float* A, float* B)
{
    float* d_A, *d_C_odd, *d_C_even;
    cudaArray* d_B;
    size_t size = sizeof(float) * ArraySize;
    int GRID_SIZE = ArraySize / BLOCK_SIZE + (ArraySize % BLOCK_SIZE != 0 ? 1 : 0);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C_odd, size);
    cudaMalloc((void**)&d_C_even, size);
    cudaMallocArray(&d_B, &SecondArrElementsRef.channelDesc, ArraySize, 1);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaBindTexture(0, FirstArrElementsRef, d_A, size);
    cudaMemcpyToArray(d_B, 0, 0, B, size, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(SecondArrElementsRef, d_B);

    float KernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    mult_particles_second << <GRID_SIZE, BLOCK_SIZE >> > (d_C_odd);
    cudaBindTexture(0, SumArrElementsRef, d_C_odd, size);
    float result = sum_particles_host(d_C_even, d_C_odd);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    printf("Second condition:\n");
    printf("Result: %f\n", result);
    printf("Elapsed time: %f\n", KernelTime);
    printf("\n");

    cudaUnbindTexture(SumArrElementsRef);
    cudaFree(d_C_odd);
    cudaFree(d_C_even);
    cudaFree(d_A);
    cudaFreeArray(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void exec_third_condition(float* A, float* B)
{
    float* d_A, *d_B, * d_C_odd, * d_C_even;
    size_t size = sizeof(float) * ArraySize;
    int GRID_SIZE = ArraySize / BLOCK_SIZE + (ArraySize % BLOCK_SIZE != 0 ? 1 : 0);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_odd, size);
    cudaMalloc((void**)&d_C_even, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaBindTexture(0, FirstArrElementsRef, d_A, size);
    cudaBindTexture(0, SecondArrElementsRef, d_B, size);

    float KernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    mult_particles_third << <GRID_SIZE, BLOCK_SIZE >> > (d_C_odd);
    cudaBindTexture(0, SumArrElementsRef, d_C_odd, size);
    float result = sum_particles_host(d_C_even, d_C_odd);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    printf("Third condition:\n");
    printf("Result: %f\n", result);
    printf("Elapsed time: %f\n", KernelTime);
    printf("\n");

    cudaUnbindTexture(SumArrElementsRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_odd);
    cudaFree(d_C_even);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float sum_particles_host(float* d_A_even, float* d_A_odd)
{
    bool isOdd = true;
    int arr_length = ArraySize;
    int iterations = arr_length / BLOCK_SIZE + (arr_length % BLOCK_SIZE != 0 ? 1 : 0);
    while (arr_length != 1)
    {
        int GRID_SIZE = iterations / BLOCK_SIZE + (iterations % BLOCK_SIZE != 0 ? 1 : 0);
        if (isOdd)
        {
            sum_particles << <GRID_SIZE, BLOCK_SIZE >> > (d_A_even, arr_length, iterations);
            cudaBindTexture(0, SumArrElementsRef, d_A_even, sizeof(float) * iterations);
        }
        else
        {
            sum_particles << <GRID_SIZE, BLOCK_SIZE >> > (d_A_odd, arr_length, iterations);
            cudaBindTexture(0, SumArrElementsRef, d_A_odd, sizeof(float) * iterations);
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

void generate_random_arr(float* A)
{
    for (int i = 0; i < ArraySize; i++)
    {
        A[i] = (rand() / (float)RAND_MAX) * (AbsMaxVal * 2 + 1) + -1 * AbsMaxVal;
        //printf("%f ", A[i]);
    }
    //printf("\n");
}