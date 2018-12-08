#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	assert(result == cudaSuccess);
  }
  return result;
}

__global__ void square(float* x, float* array, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) { 
     x[index] = sqrt( __ldg(&array[index]) );
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
  
  if ((threadsPerBlock % 32 != 0) || (threadsPerBlock > 1024)) {
     printf("threadsPerBlock must be multiple of 32 and less than 1024");
	 exit(1);
  }
  
  int size = N * sizeof(float);
  float *x = (float*) malloc(size);
  float *array = (float*) malloc(size);
  float *y = (float*) malloc(size);
  initArray(array, N);
  memcpy(y, array, size);
  
  clock_t tStart = clock();
  square_cpu(y, N);
  printf("Time taken by Host: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  
  float *xd;
  float *arrayd;
  checkCuda( cudaMalloc(&xd, size) );
  checkCuda( cudaMalloc(&arrayd, size) );
 
  checkCuda( cudaMemcpy(arrayd, array, size, cudaMemcpyHostToDevice) );
  
  // Call square kernel
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  
  tStart = clock();
  square<<<blocksPerGrid, threadsPerBlock>>>(xd, arrayd, N);
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
  checkCuda(cudaFree(xd));
  checkCuda(cudaFree(arrayd));
  free(x);
  free(y);
  free(array);
  return 0;
}
