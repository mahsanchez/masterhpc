/* Dot product of two vectors */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(argc,argv)
int argc;
char *argv[];
{
    long N, i, chunk_size;
	int rank, psize;
    float *x, *y, *local_x, *local_y;
    float local_dot, dot;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (argc < 2) {
     	fprintf(stderr,"Use: %s num_elem_vector\n", argv[0]);
     	exit(EXIT_FAILURE);
    }
	
	if (rank == 0) {
        N = atoi(argv[1]);
		
		/* Allocate memory for vectors */
		if((x = (float *) malloc(N*sizeof(float))) == NULL)
			printf("Error in malloc x[%d]\n",N);
		
		if((y = (float *) malloc(N*sizeof(float))) == NULL)
			printf("Error in malloc y[%d]\n",N);

		 /* Inicialization of x and y vectors*/
		for(i=0; i<N; i++){
			x[i] = (N/2.0 - i);
			y[i] = 0.0001*i;
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	
	MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	chunk_size = N/psize;
	
	local_x = (float *) malloc(chunk_size*sizeof(float));
	local_y = (float *) malloc(chunk_size*sizeof(float));
	
	MPI_Scatter(x, chunk_size, MPI_FLOAT, local_x, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(y, chunk_size, MPI_FLOAT, local_y, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Dot product operation 
	local_dot = 0.;
    for(i=0; i<chunk_size; i++)
	    local_dot += local_x[i] * local_y[i];
	
	MPI_Reduce(&local_dot, &dot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	double end = MPI_Wtime();
	
	if (rank == 0) {
		printf("Dot product = %f\n", dot);
		printf("elapsed time %f \n", end - start);
	} 
	
	free(local_x);
	free(local_y);
	
	if (rank == 0) {
		free(x);
	    free(y);
	}
	
	MPI_Finalize();

    return 0;
}



