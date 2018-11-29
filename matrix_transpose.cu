#include <stdio.h>

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
  // Number of rows and columns must be a multiple of 32
  const int rows = 8192;
  const int columns = 8192;
  const int width = rows;
  const int size = rows*columns*sizeof(float);

  dim3 dimGrid(width/TILE_DIM, width/TILE_DIM);
  dim3 dimBlock(TILE_DIM, TILE_DIM);

  float *A = (float*)malloc(size);
  float *B = (float*)malloc(size);
  float *C = (float*)malloc(size);
  
  float *dA, *dB;
  checkCuda( cudaMalloc(&dA, size) );
  checkCuda( cudaMalloc(&dB, size) );
    
  for (int j = 0; j < rows; j++)
    for (int i = 0; i < columns; i++)
      A[j*width + i] = 0.15*i + 0.1*j;
  
  for (int j = 0; j < rows; j++)
    for (int i = 0; i < columns; i++)
      C[j*width + i] = A[i*width + j];

  checkCuda( cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice) );
  
  transposeMatrix<<<dimGrid, dimBlock>>>(dB, dA, width);
  
  checkCuda( cudaDeviceSynchronize() );
  
  checkCuda( cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost) );
  
	for (int i = 0; i < rows * columns; i++)
	{
		if (B[i] != C[i]) {
		  printf("%d %f %f INVALID RESULTS \n", i, B[i], C[i]);
		}
	}	

  checkCuda( cudaFree(dA) );
  checkCuda( cudaFree(dB) );
  free(A);
  free(B);
  free(C);
}