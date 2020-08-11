#include <iostream>
#include <stdio.h>
#include <cublas_v2.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(int argc, char* argv[])
{
    float* arr = new float[10];
    for (int i = 0; i < 10; i++)
        arr[i] = i;
    float is_inside = 1.0f;
    float a = is_inside * arr[1000];
    std::cout << a << std::endl;
    std::cout << "completed!";
    return 0;
}