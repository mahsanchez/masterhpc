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
	MPI_Win xwin, ywin, dotwin; 

    if (argc < 2) {
     	fprintf(stderr,"Use: %s num_elem_vector\n", argv[0]);
     	exit(EXIT_FAILURE);
    }

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (rank == 0) {
        N = atoi(argv[1]);
		
		/* Allocate memory for vectors */
		if((x = (float *) malloc(N*sizeof(float))) == NULL)
		   printf("Error in malloc x[%d]\n", N);
	   
	   /* Allocate memory for vectors */
		if((y = (float *) malloc(N*sizeof(float))) == NULL)
		   printf("Error in malloc x[%d]\n", N);

		 /* Inicialization of x and y vectors*/
		for(i=0; i<N; i++){
			x[i] = (N/2.0 - i);
			y[i] = 0.0001*i;
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	
	if (rank == 0) { 
		MPI_Win_create(x, N*sizeof(float), sizeof(float), MPI_INFO_NULL,  MPI_COMM_WORLD, &xwin); 
		MPI_Win_create(y, N*sizeof(float), sizeof(float), MPI_INFO_NULL,  MPI_COMM_WORLD, &ywin); 
		MPI_Win_create(&dot, sizeof(float), sizeof(float), MPI_INFO_NULL,  MPI_COMM_WORLD, &dotwin);
    } 
    else { 
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(float), MPI_INFO_NULL,  MPI_COMM_WORLD, &xwin); 
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(float), MPI_INFO_NULL,  MPI_COMM_WORLD, &ywin); 
		MPI_Win_create(MPI_BOTTOM, 0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &dotwin); 
    } 
	
	MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	chunk_size = N/psize;
	local_x = (float *) malloc(chunk_size*sizeof(float));
	local_y = (float *) malloc(chunk_size*sizeof(float));
	
	MPI_Win_fence(0, xwin);	
	MPI_Get(local_x, chunk_size, MPI_FLOAT, 0, rank * chunk_size, chunk_size, MPI_FLOAT, xwin);
    MPI_Win_fence(0, xwin);	
	
	MPI_Win_fence(0, ywin);	
	MPI_Get(local_y, chunk_size, MPI_FLOAT, 0, rank * chunk_size, chunk_size, MPI_FLOAT, ywin);
    MPI_Win_fence(0, ywin);	

    // Dot product operation 
	local_dot = 0.;
    for(i=0; i< chunk_size; i++)
	    local_dot += local_x[i] * local_y[i];

	MPI_Win_fence(0, dotwin);
	MPI_Accumulate(&local_dot, 1, MPI_FLOAT, 0, 0, 1, MPI_FLOAT, MPI_SUM, dotwin);
	MPI_Win_fence(0, dotwin);
	
	double end = MPI_Wtime();	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (rank == 0) {
		printf("Dot product = %f\n", dot);
		printf("chunk_size = %d\n", chunk_size);
		printf("elapsed time %f \n", end);
	} 
	
	free(local_x);
	free(local_y);
	
	if (rank == 0) {
		free(x);
	    free(y);
	}
		

	MPI_Win_free(&xwin);
	MPI_Win_free(&ywin);
	MPI_Win_free(&dotwin); 
	
	MPI_Finalize();

    return 0;
}



