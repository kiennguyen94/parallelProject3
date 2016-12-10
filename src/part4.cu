#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define NUM_THREADS 256
using namespace std;
// Exclusive prescan
// n - number of elements
__global__ void prescan (int* in, int* out, int n){
	// Shared memory portion
	// Load data into shared memory for better locality
	__shared__ int temp[4*NUM_THREADS];
	// extern __shared__ int temp[];

	int tid_global = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int offset = 1;

	// temp[2*tid] = in[2*tid];
	// temp[2*tid + 1] = in[2*tid + 1];

	int a = tid;
	int b = tid + n/2;

	int bank_offsetA = CONFLICT_FREE_OFFSET(a);
	int bank_offsetB = CONFLICT_FREE_OFFSET(b);

	// temp[a+bank_offsetA] = in[a];
	// temp[b+bank_offsetB] = in[b];
	temp[a+bank_offsetA] = in[tid_global];
	temp[b+bank_offsetB] = in[tid_global + n/2];

	/*----------------------------------------------*/

	// Phase 1, up sweep 

	// >> is left shift operator, 
	// Left shift by 1 is divided by 2
	for (int i = n>>1; i>0; i >>= 1){
		__syncthreads();
		if (tid < i){
			// int a = offset*(2*tid+1)-1;
			// int b = offset*(2*tid+2)-1;

			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			a += CONFLICT_FREE_OFFSET(ai);
			b += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	/*----------------------------------------------*/

	// Phase 2, down sweep

	// if (tid==0){
	// 	temp[n-1] = 0;
	// }

	if(tid==0){
		temp[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0;
	}

	for (int d=1; d<n; d *= 2){

		// offset:= offset/2
		offset >>= 1;
		__syncthreads();

		if(tid < d){
			// int a = offset*(2*tid+1)-1;
			// int b = offset*(2*tid+2)-1;

			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			ai += CONFLICT_FREE_OFFSET(a);
			bi += CONFLICT_FREE_OFFSET(b);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// out[2*tid] = temp[2*tid];
	// out[2*tid+1] = temp[2*tid+1];

	out[tid_global] = temp[a+bank_offsetA];
	out[tid_global + n/2] = temp[b+bank_offsetB];
}

__global__ void add_base_gpu(int* device_input, int* device_output, int block_index) {
	int block_last_element = block_index * NUM_THREADS * 2 - 1;

	int base = device_input[block_last_element] + device_output[block_last_element];

	int thid = block_index * blockDim.x + threadIdx.x;

	device_output[2 * thid] += base;
	device_output[2 * thid + 1] += base;
}

void fillarray(int* in, int size){
	srand(3);
	for (int i = 0; i < size; i++){
		// in[i] = i+1;
		// in[i] = rand()%100-50;
		in[i] = rand()%2;
		//TODO put randomizer here
	}
}

template<typename T>
void printarr(T in, int size){
	for (int i = 0; i<size; i++){
		printf("%5d\t", in[i]);
	}
	printf("\n");
}

// Generate true output to compare
bool ground_truth_scan(int *in, int *out, int size){
	int temp = 0;
	for (int i = 0; i<size; i++){
		if (out[i] != temp){
			return false;
		}
		temp+=in[i];
	}
	return true;
}

bool check_mark_repeat(int* in, int* flag, int size){
	for (int i = 0; i < size-1; i++){
		if((in[i] == in[i+1]) && (flag[i] == 1)){
			continue;
		}
		else if ((in[i] == in[i+1]) && (flag[i] == 0)){
			return false;
		}
		else if ((in[i] != in[i+1]) && (flag[i] == 1)){
			return false;
		}
		else if((in[i] != in[i+1]) && (flag[i] == 0)){
			continue;
		}
	}
	return true;
}

// Mark where the duplicate happens
// The function puts 1 wherever the input array has duplicate
__global__ void mark_repeat(int *in, int *out, int size, bool repeat){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int s[2*NUM_THREADS];
	// extern __shared__ int s[];

	// Load to shared memory
	s[threadIdx.x] = in[tid];
	s[threadIdx.x+1] = in[tid+1];
	__syncthreads();
	if(tid < size -1){
		// if (in[tid] == in[tid+1]){  
		// 	out[tid] = repeat ? 1 : 0;
		// }
		// else{
		// 	out[tid] = repeat ? 0 : 1;
		// }
		if(s[threadIdx.x] == s[threadIdx.x + 1]){
			out[tid] = repeat ? 1:0;
		}
		else{
			out[tid] = repeat? 0:1;
		}
	}
	if(tid == size -1 ){
		out[tid] = 0;
	}
}

__global__ void find_repeat(int *in, int *flag, int *out, int size, bool repeat){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;

	__shared__ int temp [2*NUM_THREADS];
	__shared__ int temp2 [2*NUM_THREADS];

	temp[id] = flag[tid];
	temp[id+1] = flag[tid+1];

	temp2[id] = in[tid];

	if((tid < size - 1) && (temp[id] < temp[id+1])){
		out[temp[id]] = repeat ? tid : temp2[id];
	}
	if((tid == size-1) && (!repeat)){
		out[temp[id]] = in[size-1];
	}
}

void find_repeat_cpu(int *in, int * flag, vector<int>& output, int size, bool repeat){
	for (int i = 0; i < size-1; i++){
		if (flag[i] < flag[i+1]){
			output.push_back(repeat ? i : in[i]);
		}
	}
	if (!repeat){
		output.push_back(in[size-1]);
	}
}

void fillTest(int* in, int size){
	in[0]=1;
	in[1]=1;
	in[2]=2;
	in[3]=3;
	in[4]=4;
	in[5]=5;
	in[6]=5;
	in[7]=6;
	in[8]=6;
	in[9]=6;
	in[10]=3;
	in[11]=3;
	in[12]=4;
	in[13]=4;
	in[14]=5;
	in[15]=5;
}

int main(int argc, char** argv){
	// printf("%d\n", sizeof(int)*1024);
	// Utilities objects
	cudaEvent_t start, stop;
	cudaError_t err;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Parse input
	if (argc != 2){
		printf("Usage: ./run num_element\n");
		return 0;
	}

	// Number of elements
	int numel = atoi(argv[1]);
	// Input array
	int * arr = (int*) malloc(sizeof(int) * numel);
	// int arr[numel] = {1,1,2,3,4,5,5,6};

	// Output arr
	int * out = (int*) malloc(sizeof(int) * numel);
	// True output
	// int * out_true = (int*) malloc(sizeof(int) * numel);

	int * is_dup = (int*) malloc(sizeof(int) * numel);

	int * is_dup_scanned = (int*) malloc(sizeof(int) * numel);

	// Fill up with numbers 
	// fillarray(arr, numel);
	fillTest(arr, numel);

	// Allocate memory on device
	int *arr_d, *out_d, *is_dup_d, *is_dup_scanned_d;
	err = cudaMalloc((void**) &arr_d, sizeof(int)*numel);
	err = cudaMalloc((void**) &out_d, sizeof(int)*numel);

	// Only run on 1 block
	dim3 threads = dim3(numel, 1);
	dim3 grids(1,1);

	// Copy from host to device
	err = cudaMemcpy(arr_d, arr, sizeof(int)*numel, cudaMemcpyHostToDevice);

  	int num_block = numel / (NUM_THREADS * 2);
	if (num_block == 0) num_block = 1;
	int num_block_total = numel / NUM_THREADS;
	if (num_block_total == 0) num_block_total = 1;
  
  	/*------------------------------------------------------------------------*/
	// calculate prescan of original array

	// Launch prescan on original array
	prescan<<<num_block, NUM_THREADS>>>(arr_d, out_d, numel/num_block);
	cudaThreadSynchronize();
	for (int i = 1; i < num_block; i++){
		add_base_gpu<<<1, NUM_THREADS>>>(arr_d, out_d, i);
	}

	// Copy back to host
	err = cudaMemcpy(out, out_d, sizeof(int)*numel, cudaMemcpyDeviceToHost);

	printf("Check prescan correctness %d\n", ground_truth_scan(arr, out, numel));
	printf("Numer of block %d\n", num_block);
	cudaFree(out_d);

	// /*------------------------------------------------------------------------*/
	// // Find and mark repeat
	err = cudaMalloc((void**) &is_dup_d, sizeof(int)*numel);
	err = cudaMalloc((void**) &is_dup_scanned_d, sizeof(int)*numel);

	// Call kernel to mark repeat
	mark_repeat<<<num_block_total, NUM_THREADS>>>(arr_d, is_dup_d, numel, true);

	// Copy back to host to check
	err = cudaMemcpy(is_dup, is_dup_d, sizeof(int) * numel, cudaMemcpyDeviceToHost);
	printf("Check mark_repeat correctness %d\n", check_mark_repeat(arr, is_dup, numel));

	// Run prescan on is_dup
	prescan<<<num_block, NUM_THREADS>>>(is_dup_d, is_dup_scanned_d, numel/num_block);
	cudaThreadSynchronize();
	for (int i = 1; i < num_block; i++){
		add_base_gpu<<<1, NUM_THREADS>>>(is_dup_d, is_dup_scanned_d, i);
	}

	err = cudaMemcpy(is_dup_scanned, is_dup_scanned_d, sizeof(int) * numel, cudaMemcpyDeviceToHost);

	vector<int> duplicate_index;
	find_repeat_cpu(arr, is_dup_scanned, duplicate_index, numel, true);

	for(auto&& i : duplicate_index){
		cout<<i<<"\t";
	}

	printf("\n");
	// err = cudaMemcpy(is_dup, is_dup_d, sizeof(int)*numel, cudaMemcpyDeviceToHost);
	// err = cudaMemcpy(is_dup_scanned, is_dup_scanned_d, sizeof(int) * numel, cudaMemcpyDeviceToHost);

	// printarr (is_dup_scanned, numel);
	// Free up memory
	free(arr);
	free(out);
	free(is_dup);
	free(is_dup_scanned);
	cudaFree(is_dup_d);
	cudaFree(is_dup_scanned_d);
	cudaFree(arr_d);
	// cudaFree(out_d);
	return 0;
}