#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void checkResults(float *A, float *B, int width) {
   float maxError = 0.0f;
   for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
	      int index = i*width + j;
		  //printf("A=%f B=%f \n", A[index], B[index]);
	      maxError = fmax(maxError, fabs(A[index] - B[index] - 3.0f));
	  }
   }
   printf("Max error: %f \n", maxError );
   if (maxError != 3.0f) {
      printf("Unsuccessful results\n");
   }
   else {
      printf("Successful results\n");
   }
}



__global__ void dgemm(float *M, float *N, float *P, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < width) && (col < width)) {
		float pVal = 0;
		for (int i = 0; i < width; i++) {
		    pVal = pVal + M[row * width + i] * N[col + i*width]; 
		}
		P[row * width + col] = pVal;
	}
}


void dgemm_cpu(float *M, float *N, float *P, int width) {
    for (int row = 0; row < width; row++) 
	   for (int col = 0; col < width; col++) {
	       float pVal = 0;  
		   for (int k = 0; k < width; k++) {
				pVal = pVal + M[row * width + k] * N[col + k*width]; 
		   }
		   P[row * width + col] = pVal;
	    }	   
}

  
int main(int argc, char **argv)
{
   int width = (argc > 1)?atoi (argv[1]) : 256;
  
  const int mem_size = width*width*sizeof(float);
  int blockSize = 32; 
  int numBlocks = (width + blockSize - 1) / blockSize; 

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(numBlocks, numBlocks, 1);

  float *A_h;
  float *B_h;
  float *C_h;
  float *S_h;
  float *O_h;
  
  cudaMallocHost(&A_h, mem_size);
  cudaMallocHost(&B_h, mem_size);
  cudaMallocHost(&C_h, mem_size);
  cudaMallocHost(&S_h, mem_size);
  cudaMallocHost(&O_h, mem_size);
    
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < width; i++) {
	  int index = j*width + i;
      A_h[index] = 2;
	  B_h[index] = 2;
	  C_h[index] = 2;
	}  
  }	
  
  clock_t tStart = clock();
  dgemm_cpu(A_h, B_h, S_h, width);
  dgemm_cpu(S_h, C_h, O_h, width);
  printf("Time taken by Host: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  
  float *A_d;
  float *B_d;
  float *C_d;
  float *O_d;
  float *S_d;
  
  checkCuda( cudaMalloc(&A_d, mem_size) );
  checkCuda( cudaMalloc(&B_d, mem_size) );
  checkCuda( cudaMalloc(&C_d, mem_size) );
  checkCuda( cudaMalloc(&S_d, mem_size) );
  checkCuda( cudaMalloc(&O_d, mem_size) );
  
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1); 
  cudaStreamCreate(&stream2);
  
  checkCuda( cudaMemcpyAsync(A_d, A_h, mem_size, cudaMemcpyHostToDevice, stream1) );
  checkCuda( cudaMemcpyAsync(B_d, B_h, mem_size, cudaMemcpyHostToDevice, stream1) );
  
  tStart = clock();
  
  dgemm<<<dimGrid, dimBlock, 0, stream1>>>(A_d, B_d, S_d, width);
  checkCuda( cudaMemcpyAsync(C_d, C_h, mem_size, cudaMemcpyHostToDevice, stream2) );
  cudaDeviceSynchronize();   
  
  dgemm<<<dimGrid, dimBlock, 0, stream2>>>(S_d, C_d, O_d, width);
  cudaDeviceSynchronize();  
  clock_t tEnd = clock();
  
  checkCuda( cudaMemcpy(S_h, O_d, mem_size, cudaMemcpyDeviceToHost) );
  
  printf("Time taken by GPU: %.6fs\n", (double)(tEnd - tStart) / CLOCKS_PER_SEC);
  
  checkResults(O_h, S_h, width);

error_exit:
  cudaStreamDestroy(stream1); 
  cudaStreamDestroy(stream2);
  checkCuda( cudaFree(A_d) );
  checkCuda( cudaFree(B_d) );
  checkCuda( cudaFree(C_d) );
  checkCuda( cudaFree(O_d) );
  checkCuda( cudaFreeHost(C_h) );
  checkCuda( cudaFreeHost(A_h) );
  checkCuda( cudaFreeHost(B_h) );
  checkCuda( cudaFreeHost(S_h) );
  checkCuda( cudaFreeHost(O_h) );
}