#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores
*/

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}

__global__ void square(float *array, int k, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  
  for (int i = 0; i < k; i++) {
     int idx = index + i*stride;
     if (idx < n) {
	    array[index + i*stride] = sqrt(array[index + i*stride]);
		//printf("blockIdx.x=%d , blockDim.x=%d , ti=%d, index=%d, i=%d\n", blockIdx.x, blockDim.x, threadIdx.x, index, idx);
	 }
  }
}

void square_cpu(float *array, int n) {
   for (int i = 0; i < n; i++) {
       array[i] = sqrt(array[i]);
   }
}

void initArray(float *a, int size){
  int i;
  for(i=0; i<size; i++){
    a[i] = i*0.01;
  }
}

int main(int argc, char**argv) {
  int N = atoi(argv[1]);
  int threadsPerBlock = atoi(argv[2]);
  int k = atoi(argv[3]);
  
  if ((threadsPerBlock % 32 != 0) || (threadsPerBlock > 1024)) {
     printf("threadsPerBlock must be multiple of 32 and less than 1024");
	 exit(1);
  }
  
  int size = N * sizeof(float);
  float *x = (float*) malloc(size);
  float *y = (float*) malloc(size);
  initArray(x, N);
  memcpy(y, x, size);
  
  clock_t tStart = clock();
  square_cpu(y, N);
  printf("Time taken by Host: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  
  float *xd;
  checkCuda( cudaMalloc(&xd, size) );
 
  checkCuda( cudaMemcpy(xd, x, size, cudaMemcpyHostToDevice) );
  
  // Call square kernel
  int blocksPerGrid = (N/k + threadsPerBlock - 1)/threadsPerBlock; 
  
  tStart = clock();
  square<<<blocksPerGrid, threadsPerBlock>>>(xd, k, N);
  cudaDeviceSynchronize();
  printf("Time taken by GPU: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  checkCuda( cudaMemcpy(x, xd, size, cudaMemcpyDeviceToHost) );

  // Error Checking
  for (int i = 0; i < N; i++) {
	  if (x[i] != y[i]) {
		  printf("%d %f %f INVALID RESULTS \n", i, x[i], y[i]);
		  goto finalize;
      }
  }	
  
  printf("Successfull Sum\n");

finalize:
  free(x);
  free(y);
  checkCuda(cudaFree(xd));
  return 0;
}
