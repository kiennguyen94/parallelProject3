#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16

// Matrix multiplication with CUDA, using shared memory
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
	srand(10);
	for (int i = 0; i < size; i++){
		// in[i] = (rand() % 9)-4;
		in[i] = (float)rand()/(float)(RAND_MAX);
	}
}

// Matrix multiplication with cublas
void matmulCUBLAS(float* a, float* b, float* c, int m, int k, int n){
	int lda=m, ldb=k, ldc=m;
	const float alf = 1.0;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, m, b, k, beta, 
		c, m);
	cublasDestroy(handle);
}

void printMatrix(float* in, int width, int size){
	printf("%5.3f,", in[0]);
	for (int i = 1; i < size; i++){
		printf("%5.3f,", in[i]);
		if (i%width==0){
			printf("\n");
		}
	}
	printf ("\n");
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
	if((a_col != b_row) || (a_row % 16 != 0) || (a_col % 16 != 0) ||
		(b_row % 16 != 0) || (b_col % 16 != 0)){
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
	float *out_h_cublas = (float*) malloc(sizeof(float)*out_size);

	// Fill up with random values
	fill_rand(a_h, a_size);
	fill_rand(b_h, b_size);

	// Allocate memory on device
	float *a_d, *b_d, *out_d, *out_d_cublas;
	err = cudaMalloc((void**) &a_d, sizeof(float)*a_size);
	err = cudaMalloc((void**) &b_d, sizeof(float)*b_size);
	err = cudaMalloc((void**) &out_d, sizeof(float)*out_size);

	// Copy to device
	err = cudaMemcpy(a_d, a_h, sizeof(float)*a_size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(b_d, b_h, sizeof(float)*b_size, cudaMemcpyHostToDevice);

	// Set up block and thread size
	// Assume square block
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	// grids fits the output matrix
	dim3 grids(b_col/BLOCK_SIZE, a_row/BLOCK_SIZE);

	// Execute kernel
	cudaEventRecord(start,NULL);
	matmulGPU<<<grids, threads>>>(out_d, a_d, b_d, a_col, b_col);
	cudaEventRecord(stop,NULL);

	cudaEventSynchronize(stop);

	// Copy back the result to host
	err = cudaMemcpy(out_h, out_d, sizeof(float)*out_size, cudaMemcpyDeviceToHost);

	// Print out time
	float milliseconds = 0; 
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Kernel time %f\n", milliseconds);

	// Free memory of output matrix
	cudaFree(out_d);

	/*--------------------------------------------------------------------*/

	// CUBLAS 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	err = cudaMalloc((void**) &out_d_cublas, out_size*sizeof(float));

	// Execute
	cudaEventRecord(start, NULL);
	matmulCUBLAS(a_d, b_d, out_d_cublas, a_row, a_col, b_col);
	cudaEventRecord(stop, NULL);

	// Copy back to host
	err = cudaMemcpy(out_h_cublas, out_d_cublas, sizeof(float)*out_size, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	// Print out time
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CUBLAS time %f\n", milliseconds);

	/*--------------------------------------------------------------------*/

	// Check results
	float cumulated_err = 0;
	for (int i=0; i<out_size; i++){
		float h_temp = out_h[i];
		// printf("%d %f\n", i, h_temp);
		float h_blas_temp = out_h_cublas[i];
		// printf("%d %f\n", i, h_blas_temp);
		cumulated_err += fabs(h_temp - h_blas_temp);
	}
	printf("Normalized total error %f\n", cumulated_err/float(out_size));

	/*--------------------------------------------------------------------*/

	// Print a, b and out
	printf("Matrix a\n");
	printMatrix(a_h, a_col, a_size);
	printf("Matrix b\n");
	printMatrix(b_h, b_col, b_size);
	printf("Output\n");
	printMatrix(out_h, b_col, out_size);
	printf("Output cublas\n");
	printMatrix(out_h_cublas, b_col, out_size);
	/*--------------------------------------------------------------------*/

	cudaFree(out_d_cublas);
	free(a_h);
	free(b_h);
	free(out_h);
	free(out_h_cublas);
	cudaFree(a_d);
	cudaFree(b_d);
	int i = (a_row*a_col + b_row*b_col)*sizeof(float);
	printf("%d\n",i);
}