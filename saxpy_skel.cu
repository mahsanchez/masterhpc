/* 
   Programación de arquitecturas heterogéneas (Heterogeneous programming) 
   Unit)

   Margarita Amor 
   Jorge González

   2018

   saxpy_skel.c

   saxpy (BLAS) kernel: y<-ax+y
   Optional paramemters (in this order): saxpy #rep #n #blk
   #rep: number of executions
   #n: number of elements for each vector
   #blk: threads per CUDA block
*/

#include <stdio.h>
#include <stdlib.h>

// Type of the elements in the vector
#ifdef _INT_
typedef int basetype;     // Type for the elements: int
#define labelelem    "ints"
#elif _DOUBLE_
typedef double basetype;  // Type for the elements: double
#define labelelem    "doubles"
#else
typedef float basetype;   // Type for the elements: float     BY DEFAULT
#define labelelem    "floats"
#endif

const int N = 1048576;    // Number of elements in the vector by default

const int REP = 1;        // Number of executions by default

const int CUDA_BLK = 64;  // Number of threads in the block by default

const basetype SCALAR = (basetype) 7.0;


/* 
   In order to measure the elapsed time:

   resnfo: datatype defined to abstract the metric of the resources to use
   timenfo: datatype defined to abstract the time metric to use

   timestamp: it abstract the function used to take the time

   printtime: it abstracts the function used to print the time

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): function to obtain the time between two measures
*/

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(resnfo start, resnfo end, timenfo *t)
{
#ifdef _noWALL_
  t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) 
    - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
  t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) 
    - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
#else
  *t = (end.tv_sec + (end.tv_usec * 1E-6)) 
    - (start.tv_sec + (start.tv_usec * 1E-6));
#endif /*_noWALL_*/
}


/*
  Function to initialize the vectors
*/
void populating_arrays(basetype arrayA[], basetype arrayB[], 
		       const unsigned int n)
{
  unsigned int i;

  for(i = 0; i < n; i++) {
    arrayA[i] = i;
    arrayB[i] = n-i;
  }
}


/*
  Function to return the sum of all the elements in a vector, 
  we will use it for checking. It also initializes the array.
*/
basetype checkini_array(basetype array[], const unsigned int n)
{
  unsigned int i;
  basetype res = 0;

  for(i = 0; i < n; i++) {
    res += array[i];
    array[i] = n-i;
  }

  return(res);
}


/*
  Sequential function: saxpy for CPU (*r* times)
*/
void saxpy_CPU(const basetype arrayX[], basetype arrayY[], 
	       const basetype alpha, const unsigned int n, 
	       const unsigned int r)
{
    for (int k = 0; k < r; k++) {
		for (int i = 0; i < n; i++) {
		  arrayY[i] = alpha*arrayX[i] + arrayY[i];
		}
	}	
}


// Kernel declaration, definition below
__global__ void saxpy_kernel_cuda(const basetype *const mX, 
				  basetype *const mY, 
				  const basetype alpha, const int n);


/*
  Function to add two vectors in the GPU *r* times
*/
void saxpy_GPU(const basetype arrayX[], basetype arrayY[], 
	       const basetype alpha, const unsigned int n, 
	       const unsigned int r, const unsigned int blk_size, 
	       resnfo *const start, resnfo *const end)
{
  unsigned int t;

  // Nmber of bytes for each vector
  unsigned int numBytes = n * sizeof(basetype);

  // Allocate the global memory of the device (GPU) for our arrays and copy them
  basetype *mX, *mY;
  cudaMalloc(&mX, n*sizeof(basetype)); 
  cudaMalloc(&mY, n*sizeof(basetype));
  
  // Move input: CPU -> GPU
  cudaMemcpy(mY, arrayY, n*sizeof(basetype), cudaMemcpyHostToDevice);
  cudaMemcpy(mX, arrayX, n*sizeof(basetype), cudaMemcpyHostToDevice);

  // Unidimensional block of threads (*blk_size* threads)
  int blockSize = blk_size;

  // Unidimensional grid (*ceil(n/blk_size)* blocks)
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Call the kernel *r* times
  timestamp(start);            // Measure the calculation time on the GPU
  for (t = 0; t < r; t++) {
	  saxpy_kernel_cuda<<<numBlocks, blockSize>>>(mX, mY, alpha, n);
  }
  cudaThreadSynchronize();
  timestamp(end);

  // Move result: GPU -> CPU
  cudaMemcpy(arrayY, mY, n*sizeof(basetype), cudaMemcpyDeviceToHost);

  // Free the global memory in the device
  cudaFree(mX);
  cudaFree(mY);
}


// Declaration of the function to see the device resources
void devicenfo(void);


// Declaration of the function to check and adjust the execution parameters of the kernel to the restrictions of the GPU
void checkparams(unsigned int *n, unsigned int *cb);


/*
  Main function
*/
int main(int argc, char *argv[])
{
  // To measure time
  resnfo start, end, startgpu, endgpu;
  timenfo time, timegpu;

  // Accept some parameters

  // Number of times to repeat the operation (by default: REP)
  unsigned int reps = (argc > 1)?atoi (argv[1]):REP;

  if (reps == 0) {
    devicenfo();
    return(0);
  }

  // Number of elements in the vectors (by default: N)
  unsigned int n = (argc > 2)?atoi (argv[2]):N;

  // Number of threads per block (by default: CUDA_BLK)
  unsigned int cb = (argc > 3)?atoi (argv[3]):CUDA_BLK;

  checkparams(&n, &cb);

  // Number of bytes to allocate in our vectors
  unsigned int numBytes = n * sizeof(basetype);

  // Allocate and initialize vectorss
  timestamp(&start);
  basetype *vectorA = (basetype *) malloc(numBytes);
  basetype *vectorB = (basetype *) malloc(numBytes);
  populating_arrays(vectorA, vectorB, n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Allocate and initialize vectors (%u %s)\n\n", n, labelelem);


  // Execute SAXPY on the CPU
  timestamp(&start);
  saxpy_CPU(vectorA, vectorB, SCALAR, n, reps);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> saxpy on CPU (%u times)\n\n", reps);

  // Add the elements of the result vector, to ckeck the calculation on the CPU
  basetype result = checkini_array(vectorB, n);

  //
  populating_arrays(vectorA, vectorB, n);
  
  // Execute saxpy in GPU
  timestamp(&start);
  saxpy_GPU(vectorA, vectorB, SCALAR, n, reps, cb, &startgpu, &endgpu);
  timestamp(&end);

  // Add the elements of the result vector, to check the calculation on the GPU
  basetype result_gpu = checkini_array(vectorB, n);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> saxpy on GPU (%d threads, %d threads/block, %u times)\n", n, cb, reps);
  if (result_gpu == result) // Comprobamos si resultado numérico es OK
    printf("\t\t      Result of the addition OK\n\n");
  else
    printf("\t\t      mec!\n\n");

  // We distinguish between calculation time on the GPU and transfer time
  myElapsedtime(startgpu, endgpu, &timegpu);	
  printtime(timegpu);
  printf("calculation time on the GPU\n\t\t%15f s alloc and comm\n", time - timegpu);

  free(vectorA);
  free(vectorB);

  return(0);
}


/*
  Definition of our kernel for axpy
*/
__global__ void saxpy_kernel_cuda(const basetype *const mX, 
				  basetype *const mY, 
				  const basetype alpha, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
	   mY[i] = alpha*mX[i] + mY[i];
	}   
}


/*
  Sacar por pantalla información del *device*
*/
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


/*
  Function to adjust the number of threads, blocks, and blocks per thread according to the restrictions of the GPU
*/
void checkparams(unsigned int *n, unsigned int *cb)
{
  struct cudaDeviceProp capabilities;

  // We have to reduce the block
  if (*cb > *n)
    *cb = *n;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Number of threads/block modified to %d (max per block for dev)\n\n", 
	   *cb);
  }

  if (((*n + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (*n - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Number of threads/block modified to %d (max per block for dev)\n", 
	     *cb);
      if (*n > (capabilities.maxGridSize[0] * *cb)) {
	*n = capabilities.maxGridSize[0] * *cb;
	printf("->Total number of threads modified to %d (max per grid for dev)\n\n", *n);
      } else {
	printf("\n");
      }
    } else {
      printf("->Number of threads/block modified to %d (%d max block/grid for dev)\n\n", 
	     *cb, capabilities.maxGridSize[0]);
    }
  }
}
