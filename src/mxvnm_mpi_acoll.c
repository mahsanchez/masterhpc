 /* Matrix-vector product. Matrix of N*M */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc,char *argv[])
{
    int i, j, N, M, dim[2];
	int rank, psize;
    float *A, *x, *y, temp;
	MPI_Request request;
	MPI_Status status;

    if (argc < 3) {
     	fprintf(stderr,"Usar: %s filas columnas\n", argv[0]);
     	exit(EXIT_FAILURE);
    }
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
			N = atoi(argv[1]);  //Rows of the matrix and elements of vector y
			M = atoi(argv[2]);  //Columns of the matrix and elements of vector x
			dim[0] = M;
			dim[1] = N;

			/* Allocate memory for global matrix */
			if((A = (float *) malloc(N*M*sizeof(float))) == NULL)
			   printf("Error en malloc Avector[%d]\n",N*M);

			/* Matrix inicialization */
		    for(i=0; i < N; i++)
			   for(j=0; j < M; j++)
				  A[i*N + j] = (0.15*i - 0.1*j)/N;
			  			
            /* Allocate memory for x and y vectors */
			//if((x = (float *) malloc(M*sizeof(float))) == NULL)
			//	printf("Error en malloc x[%d]\n",M);
			
			if((y = (float *) malloc(N*sizeof(float))) == NULL)
				printf("Error en malloc y[%d]\n",N);

			/* x vector inicialization */
			/* for(i=0; i<M; i++)
			   x[i] = (M/2.0 - i); 	*/		
	}
	
    MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	
	MPI_Bcast(dim, 2, MPI_INT, 0, MPI_COMM_WORLD);
	
	int local_M = dim[0];
	int local_N = dim[1]/psize;
	
	// Initialize vector x locally on each process and distribute it among the process
	float *local_x = (float *) malloc( local_M/psize * sizeof(float));
	x = (float *) malloc( dim[0] * sizeof(float) );
	
	for (i = 0; i < local_M/psize; i++) {
		local_x[i] = (local_M/2.0 - 1);
	}
	
	MPI_Iallgather(local_x, local_M/psize, MPI_FLOAT, x, local_M/psize, MPI_FLOAT, MPI_COMM_WORLD, &request);
	MPI_Wait(&request, &status);
	
	// Matrix distribution among process
	int chunk_size = local_M * local_N;
	float *local_A = (float *) malloc(chunk_size * sizeof(float));
    MPI_Iscatter(A, chunk_size, MPI_FLOAT, local_A, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD, &request);
	MPI_Wait(&request, &status);
	
	// local_y partial results vector
	float *local_y = (float *) malloc( local_N * sizeof(float));

    // Matrix-vector product, y = Ax   
    for(i=0; i < local_N; i++) {
		temp = 0.0;
		for(j=0; j < local_M; j++)
			temp += local_A[i*local_N + j] * x[j];				
		local_y[i] = temp;
    }
	
	MPI_Igather(local_y, local_N, MPI_FLOAT, y, local_N, MPI_FLOAT, 0, MPI_COMM_WORLD, &request);
	MPI_Wait(&request, &status);
	
	double end = MPI_Wtime();	
	    
	if (rank == 0) {
		printf("Done,  y[0] = %4.2f  y[%d] = %4.2f \n", y[0], N-1, y[N-1]);
		printf("dimensions M=%d N=%d \n", dim[0], dim[1]);
		printf("elapsed time %f \n", end - start);
	}	
	
	if (rank == 0) {
		free(A); 
		free(y);
	}
	
	free(x);
	free(local_x);
	free(local_y);
	
	MPI_Finalize();

    return 0;
}



