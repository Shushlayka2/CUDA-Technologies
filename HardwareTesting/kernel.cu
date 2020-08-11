#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <stdio.h>
#include <ctime>
#include <fstream>

using namespace std;

vector<char> add_data(vector<char> data);
void copy_data_to_gpu(char* data, int size);

int main()
{
	int size = 0;
	vector<char> data;
	for (int i = 1; i <= 50; i++)
	{
		data = add_data(data);
		size += 1048576;
		copy_data_to_gpu(&data[0], size);
	}
	data.clear();
	data.shrink_to_fit();
}

vector<char> add_data(vector<char> data)
{
	int random_integer;
	for (int i = 0; i < 1048576; i++)
	{
		random_integer = rand() % 10;
		data.push_back(random_integer);
	}
	return data;
}

void copy_data_to_gpu(char* data, int size)
{
	ofstream outfile;
	outfile.open("result.txt", ofstream::out | ofstream::app);
	outfile.seekp(0, ios::end);

	char* dev_data;
	int sumTo = 0, sumFrom = 0;
	for (int i = 0; i < 100; i++)
	{
		int elapsedTime;
		cudaMalloc((void**)&dev_data, size);
		clock_t begin = clock();
		cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_t end = clock();
		elapsedTime = double(end - begin);
		sumTo += elapsedTime;

		begin = clock();
		cudaMemcpy(data, dev_data, size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		end = clock();
		elapsedTime = double(end - begin);
		sumFrom += elapsedTime;

		cudaFree(dev_data);
	}
	outfile << sumTo / (float)100 << " " << sumFrom / (float)100 << endl;
	outfile.close();
}