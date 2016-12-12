#include <iostream>


__device__ inline double __shfl_down(double var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}

__inline__ __device__ int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__ int blockReduceSum(int val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void deviceReduceKernel(int *in, int* out, int N) {
  int sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

void deviceReduce(int *in, int* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

void fillarray(int* in, int size){
	srand(3);
	for (int i = 0; i < size; i++){
		// in[i] = i+1;
		// in[i] = rand()%100-50;
		in[i] = rand()%100;
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


int main(int argc, char** argv){
	cudaError_t err;

	//TODO: read input

	int numel = 16;
	int *arr = (int*) malloc(sizeof(int) * numel);
	int *out = (int*) malloc(sizeof(int) * numel);

	fillarray(arr, numel);

	// Allocate device memory
	int *arr_d, *out_d;
	err = cudaMalloc((void**) &arr_d, sizeof(int)*numel);
	err = cudaMalloc((void**) &out_d, sizeof(int)*numel);

	// Copy to device
	err = cudaMemcpy(arr_d, arr, sizeof(int)*numel, cudaMemcpyHostToDevice);

	// Do reduction
	deviceReduce(arr_d, out_d, numel);

	// Copy back to host
	err = cudaMemcpy(out, out_d, sizeof(int)*numel, cudaMemcpyDeviceToHost);
	printarr(out, numel);

	cudaFree(arr_d);
	cudaFree(out_d);
	free(arr);
	free(out);

	return 0;
}