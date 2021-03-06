#include <stdio.h>
#include <time.h>

/*
Measure Time
Maximum Matrix Size 
*/

const int TILE_DIM = 32;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	exit(1);
  }	
  return result;
}
 


__global__ void transposeMatrix(float *B, float *A, int width)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  tile[threadIdx.y][threadIdx.x] = A[y*width + x];
  __syncthreads();
  
  x = blockIdx.y * TILE_DIM + threadIdx.x;  
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  
  B[y*width + x] = tile[threadIdx.x][threadIdx.y];
}


int main(int argc, char **argv)
{
  if (argc < 1) {
     printf("matrix size is a mandatory parameter");
	 exit(1);
  }
  
  // Number of rows and columns must be a multiple of 32
  int rows = atoi(argv[1]);
  
  if (rows % TILE_DIM ) {
    printf("number of rows must be multiple of 32\n");
    exit(1);
  }
  
  const int columns = rows;
  const int width = rows;
  const int size = rows*columns*sizeof(float);

  dim3 dimGrid(width/TILE_DIM, width/TILE_DIM);
  dim3 dimBlock(TILE_DIM, TILE_DIM);

  float *A, *B, *C;
  checkCuda( cudaMallocHost(&A, size) );
  checkCuda( cudaMallocHost(&B, size) );
  checkCuda( cudaMallocHost(&C, size) );
  /*
  A = (float*)malloc(size);
  B = (float*)malloc(size);
  C = (float*)malloc(size);
  */
  
  float *dA, *dB;
  checkCuda( cudaMalloc(&dA, size) );
  checkCuda( cudaMalloc(&dB, size) );
    
  for (int j = 0; j < rows; j++)
    for (int i = 0; i < columns; i++)
      A[j*width + i] = 0.15*i + 0.1*j;
  
  clock_t tStart = clock();
  for (int j = 0; j < rows; j++)
    for (int i = 0; i < columns; i++)
      C[j*width + i] = A[i*width + j];
  printf("Time taken by Host: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

  checkCuda( cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice) );
  
  tStart = clock();
  transposeMatrix<<<dimGrid, dimBlock>>>(dB, dA, width); 
  checkCuda( cudaDeviceSynchronize() );
  printf("Time taken by GPU: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  
  checkCuda( cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost) );
  
	for (int i = 0; i < rows * columns; i++)
	{
		if (B[i] != C[i]) {
		  printf("%d %f %f INVALID RESULTS \n", i, B[i], C[i]);
		  goto finished;
		}
	}	
	
   printf("Matrix Transpose Successful");	

finished:
  checkCuda( cudaFree(dA) );
  checkCuda( cudaFree(dB) );
  checkCuda(cudaFreeHost(A));
  checkCuda(cudaFreeHost(B));
  checkCuda(cudaFreeHost(C));
}