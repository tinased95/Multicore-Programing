#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;
// N = 2^22
#define N pow(2, 22)

#define BLOCK_SIZE 64

int reduction(int *a);

__global__ void reduce4(int *g_inData, int *g_outData)
{
	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	// Fill up the shared memory
	sdata[tid] = g_inData[i] + g_inData[i + blockDim.x];
	__syncthreads();


	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// Write the result for this block to global memory
	if (tid == 0) {
		g_outData[blockIdx.x] = sdata[0];
	}

}

int main(int argc, char *argv[]) {
	// capture start time all
	auto start_time_all = chrono::high_resolution_clock::now();

	// initialize a vector of size N with 1
	vector<int> v(N, -1);
	//for (vector <int> ::iterator  i = v.begin(); i != v.end(); ++i)
	//	cout << *i << '\t';
	// reduction algorithm 1 call
	reduction(&v[0]);

	// capture start time
	auto start_time = chrono::high_resolution_clock::now();

	// reduction
	auto sum = accumulate(begin(v), end(v), 0);

	// capture end time
	auto end_time = chrono::high_resolution_clock::now();

	// elapsed time in milliseconds
	auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

	// print sum and elapsed time
	cout << "[-] Serial Sum: " << sum << endl;
	cout << "[-] Serial Duration: " << duration.count() << "ms" << endl;

	// capture end time all
	auto end_time_all = chrono::high_resolution_clock::now();
	// elapsed time all in milliseconds
	auto duration_all = chrono::duration_cast<chrono::microseconds>(end_time_all - start_time_all);
	cout << "[-] Overall Duration: " << duration_all.count() << "ms" << endl;


	system("pause");
	return EXIT_SUCCESS;
}

int reduction(int *a)
{
	// For checking errors
	cudaError_t error;

	// Define the pointers
	int *d_A;
	int *d_Blocks;

	// Define the size of arrays
	int memSize = N * sizeof(int);

	// Allocate space on GPU
	error = cudaMalloc((void **)&d_A, memSize);

	// Copy data to GPU
	error = cudaMemcpy(d_A, a, memSize, cudaMemcpyHostToDevice);


	// Calculate execution parameters
	int grid_x = N / BLOCK_SIZE;
	int block_x = BLOCK_SIZE;

	// Define the proper size for blocks array
	int blocksMemSize = grid_x * sizeof(int);

	// Allocate space on GPU 
	error = cudaMalloc((void **)&d_Blocks, memSize);

	// Setup execution parameters
	dim3 gridDimensions(grid_x, 1, 1);
	dim3 blockDimensions(block_x, 1, 1);

	printf("Computing result using CUDA Kernel...\n");
	// Create the start and stop timer
	cudaEvent_t start;
	error = cudaEventCreate(&start);
	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	// Start the timer
	error = cudaEventRecord(start, NULL);

	int toDo;
	if (block_x>1) toDo = 1 + block_x / 128;
	else toDo = 0;

	// Execute Kernels
	reduce4 << <gridDimensions, blockDimensions, block_x*sizeof(int) >> > (d_A, d_Blocks);
	//reduce4 << <gridDimensions, blockDimensions, block_x*sizeof(int) >> > (d_Blocks, d_Blocks);

	for (int i = 1; i < toDo; i++) {
		//	reduce0 << <gridDimensions, blockDimensions, block_x*sizeof(int) >> > (d_Blocks, d_Blocks);
	}

	//	printf("result %s\n", d_Blocks);
	// Kernel launch error handling
	error = cudaGetLastError();

	// Stop the timer
	error = cudaEventRecord(stop, NULL);

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	// Calculate the GPU Duration
	float elapsed_time = 0.0f;
	error = cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "[-] GPU Duration: " << elapsed_time << "ms" << endl;

	// Copy result back to host
	error = cudaMemcpy(a, d_Blocks, memSize, cudaMemcpyDeviceToHost);

	cout << "[-] GPU sum: " << *a << endl;
	printf("Effective Bandwidth (GB/s): %f \n", N * 4 * 3 / elapsed_time / 1e6);
	// Free up 
	cudaFree(d_A);
	cudaFree(d_Blocks);

	return EXIT_SUCCESS;
}