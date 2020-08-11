#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath> 
#include <stdio.h>
#include <malloc.h>

using namespace std;

#define ArrSize 20
#define BlockSize 32
#define BaseType double
#define DevOperation __exp10f//__expf//__dsqrt_rn//__fsqrt_rn
#define HostOperation(arg) exp10(arg)//exp//sqrt

#define STR_EXPAND(arg) #arg
#define STR(arg) STR_EXPAND(arg)

__global__ void calc(BaseType* A)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ArrSize)
		A[i] = DevOperation(i);
}

int main()
{
	int GridSize = ArrSize / BlockSize + (ArrSize % BlockSize != 0 ? 1 : 0);
	size_t size = sizeof(BaseType) * ArrSize;

	BaseType* A_dev;
	cudaMalloc((void**)&A_dev, size);

	float KernelTime;
	cudaEvent_t start, stop;  
	cudaEventCreate(&start);  
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	calc << <GridSize, BlockSize >> > (A_dev);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&KernelTime, start, stop);

	BaseType *A_host;
	A_host = (BaseType*)malloc(size);
	cudaMemcpy(A_host, A_dev, size, cudaMemcpyDeviceToHost);

	BaseType err = 0.0;
	for (int i = 0; i < ArrSize; i++)
	{
		printf("%.2f %.2f\n", HostOperation(i * 1.0), A_host[i]);
		err += abs(HostOperation(i * 1.0) - A_host[i]);
	}
		
	err /= ArrSize;

	printf("Used type is %s\nOperation is %s\nEllapsed time equals to %f milliseconds\nError equals to %f", 
		typeid(BaseType).name(), STR(DevOperation), KernelTime, err);

	cudaFree(A_dev);
	cudaEventDestroy(start);  
	cudaEventDestroy(stop);
	return 0;
}