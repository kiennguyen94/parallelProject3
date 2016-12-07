#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 32

__global__ void matmulGPU(float * out, float * a, float * b, int wa, int wb){
	// Get block index
	int b_col = blockIdx.x;
	int b_row = blockIdx.y;

	// Get thread index
	int t_col = threadIdx.x;
	int t_row = threadIdx.y;

	// Get submatrices of a and b, load up to shared memory
	// Index of first submatrix of a in the block
	int a_start = wa * BLOCK_SIZE * b_row;
	// Index of last submatrix of a
	int a_end = a_start + wa -1;
	int a_stride = BLOCK_SIZE;
	// Index of first submatrix of b in the block
	int b_start = BLOCK_SIZE * b_col;
	int b_stride = BLOCK_SIZE * wb;

	float temp = 0;

	// Lop through all submatrix of a and b
	for (int i = a_start, j = b_start; i <= a_end; i+=a_stride, j+=b_stride){
		// allocate on shared memory
		__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

		// a and b are a contigous array
		a_shared[t_row][t_col] = a[i + wa * t_row + t_col];
		b_shared[t_row][t_col] = b[j + wb * t_row + t_col];

		// Sync to make sure all submatrices finished loading
		__syncthreads();

		#pragma unroll
		for(int k = 0; k < BLOCK_SIZE; k++){
			temp += a_shared[t_row][k] * b_shared[k][t_col];
		}

		// Sync to make sure all threads finished calculating
		__syncthreads();
	}

	// Write back to device memory
	int out_idx = wb * BLOCK_SIZE * b_row + BLOCK_SIZE * b_col;
	out[out_idx + wb * t_row + t_col] = temp;
}

// Fill up with randomized element
void fill_rand(float *in, int size){
	for (int i = 0; i < size; i++){
		in[i] = (rand() % 9)-4;
	}
}

int main(int argc, char** argv){


	// Timing objects
	cudaEvent_t start, stop;
	cudaError_t err;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Read and parse input
	if (argc!=5){
		printf("Usage: ./run a_row a_col b_row b_col\na_col and b_row should match. All numbers should be multiple of 32");
		return 0;
	}

	int a_row = atoi(argv[1]);
	int a_col = atoi(argv[2]);
	int b_row = atoi(argv[3]);
	int b_col = atoi(argv[4]);

	// Enforce dimension
	if((a_col != b_row) || (a_row % 32 != 0) || (a_col % 32 != 0) ||
		(b_row % 32 != 0) || (b_col % 32 != 0)){
		printf("a_col and b_row should match. All numbers should be multiple of 32\n");
		return 0;
	}

	// Host memory
	unsigned int a_size = a_row * a_col;
	float *a_h = (float*) malloc(sizeof(float)*a_size);
	unsigned int b_size = b_row * b_col;
	float *b_h = (float*) malloc(sizeof(float)*b_size);
	unsigned int out_size = a_row * b_col;
	float *out_h = (float*) malloc(sizeof(float)*out_size);

	// Fill up with random values
	fill_rand(a_h, a_size);
	fill_rand(b_h, b_size);

	// Allocate memory on device
	float *a_d, *b_d, *out_d;
	err = cudaMalloc((void**) &a_d, a_size);
	err = cudaMalloc((void**) &b_d, b_size);
	err = cudaMalloc((void**) &out_d, out_size);

	// Copy to device
	err = cudaMemcpy(a_d, a_h, a_size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(b_d, b_h, b_size, cudaMemcpyHostToDevice);

	// Set up block and thread size
	// Assume square block
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	// grids fits the output matrix
	dim3 grids(b_col/BLOCK_SIZE, a_row/BLOCK_SIZE);

	// Execute kernel
	cudaEventRecord(start,NULL);
	matmulGPU<<<grids, threads>>>(out_d, a_d, b_d, a_col, b_col);
	cudaEventRecord(stop,NULL);

	err = cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0; 
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Kernel time %f\n", milliseconds);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(out_d);
	int i = (a_row*a_col + b_row*b_col)*sizeof(float);
	printf("%d\n",i);
}