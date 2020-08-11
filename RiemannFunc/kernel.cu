#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>

using namespace std;

__global__ void calc(double* arr, int s, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < n) {
		arr[x] = 1 / powf(x + 1, s);
	}
}

int main()
{
	double* arr;
	double* arr_device;
	int n = 20000;
	int s;
	cout << "Enter the s param" << endl;
	cin >> s;
	arr = (double*)malloc(sizeof(double) * n);
	cudaMalloc((void**)&arr_device, sizeof(double) * n);
	int blockSize = 5;
	int gridSize = n / blockSize + (n % blockSize == 0 ? 0 : 1);
	calc << < gridSize, blockSize >> > (arr_device, s, n);
	cudaDeviceSynchronize();
	cudaMemcpy(arr, arr_device, sizeof(double) * n, cudaMemcpyDeviceToHost);
	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += arr[i];
	}
	printf("Result equls to %0.7f\n", sum);
	return 0;
}