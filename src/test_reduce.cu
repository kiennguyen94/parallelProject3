#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <algorithm>
using namespace std;

#define BLOCK_SIZE 512

__global__ void reduce_max(float * in, float * out, int numel, float smallest) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float s[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < numel)
       s[t] = in[start + t];
    else
       s[t] = smallest;
    if (start + BLOCK_SIZE + t < numel)
       s[BLOCK_SIZE + t] = in[start + BLOCK_SIZE + t];
    else
       s[BLOCK_SIZE + t] = smallest;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          s[t] = fmax(s[t], s[t+stride]);
    }
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    if (t == 0)
       out[blockIdx.x] = s[0];
}

void fillarray(float* in, int size){
  srand(3);
  for (int i = 0; i < size; i++){
    // in[i] = i+1;
    // in[i] = rand()%100-50;
    in[i] = rand()%100;
    //TODO put randomizer here
  }
}

void fillTest(float* in, int size){
  // in[0]=1.0;
  // in[1]=1.0;
  // in[2]=2.0;
  // in[3]=3.0;
  // in[4]=4.0;
  // in[5]=5.0;
  // in[6]=5.0;
  // in[7]=6.0;
  // in[8]=6.0;
  // in[9]=6.0;
  // in[10]=3.0;
  // in[11]=3.0;
  // in[12]=4.0;
  // in[13]=4.0;
  // in[14]=5.0;
  // in[15]=5.0;
  for (int i = 0; i < size; i++){
    in[i] = i;
  }
}


int main(int argc, char ** argv) {
    int ii;
    // wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    float smallest = numeric_limits<float>::min();

    if (argc != 2){
      printf("wrong arguments\n");
      return 0;
    }

    numInputElements = atoi(argv[1]);

    hostInput = (float*) malloc(numInputElements * sizeof(float));

    fillTest (hostInput, numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }

    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    cudaMalloc(&deviceInput, sizeof(float) * numInputElements);
    cudaMalloc(&deviceOutput, sizeof(float) * numOutputElements);
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * numInputElements, cudaMemcpyHostToDevice);
    dim3 dimGrid(numOutputElements, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    reduce_max<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements, smallest);

    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);

    for (ii = 1; ii < numOutputElements; ii++) {
      hostOutput[0] = max(hostOutput[ii], hostOutput[0]);
    }

    printf("Final max %f\n", hostOutput[0]);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostInput);
    free(hostOutput);
}