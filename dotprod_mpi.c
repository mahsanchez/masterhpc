/* Dot product of two vectors */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(argc,argv)
int argc;
char *argv[];
{
    int N=100000000, i, chunk_size; 
	int myrank, provided, numprocs;
    float *x, *y, *local_x, *local_y;
    float dot, local_dot;
    
    if (argc < 2) {
     	fprintf(stderr,"Use: %s num_elem_vector\n", argv[0]);
     	exit(EXIT_FAILURE);
    }

    N = atoi(argv[1]);
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    printf("Nivel proporcionado %d de %d, %d, %d, %d\n", provided, MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
	
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	if (N % numprocs !=0) {
       if (myrank==0) printf("Vector size not divisible by number of processors\n");
       MPI_Finalize();
       exit(-1);
    }

    /* Allocate memory for vectors */
    if((x = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error in malloc x[%d]\n",N);
	
    if((y = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error in malloc y[%d]\n",N);

    /* Inicialization of x and y vectors*/
	if (myrank == 0) {
		for (i=0; i<N; i++) {
			x[i] = (N/2.0 - i);
			y[i] = 0.0001*i;
        }
	}
    
	chunk_size = N/numprocs;
	local_x = (float *) malloc(chunk_size * sizeof(float));
	local_y = (float *) malloc(chunk_size * sizeof(float));
	
	MPI_Scatter(x, chunk_size, MPI_FLOAT, local_x, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(y, chunk_size, MPI_FLOAT, local_y, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Dot product operation */  
    	
    local_dot = 0.;
	#pragma omp parallel for default(none) private(i) shared(chunk_size, local_x, local_y) reduction(+:local_dot)
    for(i=0; i < chunk_size; i++)
	    local_dot += local_x[i] * local_y[i];
	
	MPI_Reduce(&local_dot, &dot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  
    if (myrank == 0) {
		printf("Dot product = %g\n", dot);
	} 
	
	MPI_Finalize();

    return 0;
}



