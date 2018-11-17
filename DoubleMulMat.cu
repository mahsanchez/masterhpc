#include <stdio.h>
#include <assert.h>

const int NUM_REPS = 100;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}



void devicenfo(void)
{
  struct cudaDeviceProp capabilities;

  cudaGetDeviceProperties (&capabilities, 0);
  printf("->CUDA Platform & Capabilities\n");
  printf("Name: %s\n", capabilities.name);
  printf("totalGlobalMem: %.2f MB\n", capabilities.totalGlobalMem/1024.0f/1024.0f);
  printf("sharedMemPerBlock: %.2f KB\n", capabilities.sharedMemPerBlock/1024.0f);
  printf("regsPerBlock (32 bits): %d\n", capabilities.regsPerBlock);
  printf("warpSize: %d\n", capabilities.warpSize);
  printf("memPitch: %.2f KB\n", capabilities.memPitch/1024.0f);
  printf("maxThreadsPerBlock: %d\n", capabilities.maxThreadsPerBlock);
  printf("maxThreadsDim: %d x %d x %d\n", capabilities.maxThreadsDim[0], 
	 capabilities.maxThreadsDim[1], capabilities.maxThreadsDim[2]);
  printf("maxGridSize: %d x %d\n", capabilities.maxGridSize[0], 
	 capabilities.maxGridSize[1]);
  printf("totalConstMem: %.2f KB\n", capabilities.totalConstMem/1024.0f);
  printf("major.minor: %d.%d\n", capabilities.major, capabilities.minor);
  printf("clockRate: %.2f MHz\n", capabilities.clockRate/1024.0f);
  printf("textureAlignment: %d\n", capabilities.textureAlignment);
  printf("deviceOverlap: %d\n", capabilities.deviceOverlap);
  printf("multiProcessorCount: %d\n", capabilities.multiProcessorCount);
}

void checkResults(float *A, float *B, int width) {
   float maxError = 0.0f;
   for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
	      int index = i*width + j;
	      maxError = fmax(maxError, fabs(A[index] - B[index] - 3.0f));
		  printf("%f=%f \n", A[index], B[index]);
	  }
   }
   printf("Max error: %f \n", maxError );
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
   if (width == 0) {
      devicenfo();
      return(0);
  }
  
  const int mem_size = width*width*sizeof(float);
  int blockSize = 32; 
  int numBlocks = (width + blockSize - 1) / blockSize; 

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(numBlocks, numBlocks, 1);

  float *A_h = (float*)malloc(mem_size);
  float *B_h = (float*)malloc(mem_size);
  float *C_h = (float*)malloc(mem_size);
  float *D_h = (float*)malloc(mem_size);
  
  float *A_d;
  float *B_d;
  float *C_d;
  
  checkCuda( cudaMalloc(&A_d, mem_size) );
  checkCuda( cudaMalloc(&B_d, mem_size) );
  checkCuda( cudaMalloc(&C_d, mem_size) );
    
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < width; i++) {
	  int index = j*width + i;
      A_h[index] = 2;
	  B_h[index] = 2;
	}  
  }	
  
  // host
  dgemm_cpu(A_h, B_h, D_h, width);
  
  // device
  checkCuda( cudaMemcpy(A_d, A_h, mem_size, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(B_d, B_h, mem_size, cudaMemcpyHostToDevice) );
  
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  printf("%25s", "matrix multiplication");
  
  checkCuda( cudaEventRecord(startEvent, 0) );
  
  dgemm<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
  cudaDeviceSynchronize(); 	
	 
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  
  checkCuda( cudaMemcpy(C_h, C_d, mem_size, cudaMemcpyDeviceToHost) );
  
  checkResults(D_h, C_h, width);

error_exit:
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  
  checkCuda( cudaFree(A_d) );
  checkCuda( cudaFree(B_d) );
  checkCuda( cudaFree(C_d) );
  free(C_h);
  free(A_h);
  free(B_h);
  free(D_h);
}