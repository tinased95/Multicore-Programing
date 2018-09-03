#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define N 2048
#define thread_num	512
#define block_num 4

__global__ void prescan(float *g_odata, float *g_idata, int n);
void scanCPU(float *f_out, float *f_in, int i_n);

int main()
{
	float a[N], c[N], g[N];
	float *dev_a, *dev_g;
	int size = N * sizeof(float);

	double d_gpuTime, d_cpuTime;
	cudaError_t error;
	// initialize matrice a
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
		printf("a[%i] = %f\n", i, a[i]);
	}

	// initialize a and g matrices here
	error = cudaMalloc((void **)&dev_a, size);
	error = cudaMalloc((void **)&dev_g, size);

	error = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	//execute kernel
	prescan << <block_num, thread_num, 2 * thread_num*sizeof(float) >> >(dev_g, dev_a, N);
	cudaDeviceSynchronize();

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}



	cudaMemcpy(g, dev_g, size, cudaMemcpyDeviceToHost);



	clock_t cpu_startTime, cpu_endTime;
	float cpu_ElapseTime = 0;
	cpu_startTime = clock();
	scanCPU(c, a, N);

	cpu_endTime = clock();
	cpu_ElapseTime = (double)(cpu_endTime - cpu_startTime) / CLOCKS_PER_SEC;

	cudaFree(dev_a); cudaFree(dev_g);

	for (int i = 0; i < N; i++)
	{
		printf("c[%i] = %0.3f, g[%i] = %0.3f\n", i, c[i], i, g[i]);
	}

	//printf("start= %.100f msec\nend= %.100f msec\n", (float)cpu_startTime, (float)cpu_endTime);
	// Compute and print the gpu time
	///printf("GPU Time= %.3f msec\nCPU Time= %.100f msec\n", msecTotal, cpu_ElapseTime);
	//printf("CPU Time= %.100f msec\n", cpu_ElapseTime);
	printf("GPU Time= %.3f msec\n", msecTotal);

	//	printf("GPU Time for scan size %i: %f\n", N, d_gpuTime);
	//	printf("CPU Time for scan size %i: %f\n", N, d_cpuTime);
	system("PAUSE");
}


__global__ void prescan(float *g_odata, float *g_idata, int n)
{
	extern  __shared__  float temp[];
	// allocated on invocation
	int thid = threadIdx.x;
	int bid = blockIdx.x;


	int offset = 1;
	if ((bid * thread_num + thid)<n) {
		temp[thid] = g_idata[bid * thread_num + thid];
	}
	else {
		temp[thid] = 0;
	} // Make the "empty" spots zeros, so it won't affect the final result.

	for (int d = thread_num >> 1; d > 0; d >>= 1)
		// build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0)
	{
		temp[thread_num - 1] = 0;
	}

	// clear the last element
	for (int d = 1; d < thread_num; d *= 2)
		// traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_odata[bid * thread_num + thid] = temp[thid];
}

void scanCPU(float *f_out, float *f_in, int i_n)
{
	f_out[0] = 0;
	for (int i = 1; i < i_n; i++)
		f_out[i] = f_out[i - 1] + f_in[i - 1];

}
