#include "mpi.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

int main(int argc, char *argv[]) 
{ 
    long n=1000000000, i;
    int rank, psize; 
    double PI25DT = 3.141592653589793238462643; 
    double pi, local_pi, h, sum, x; 
    MPI_Win nwin, piwin; 
 
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &psize); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	
	if (argc < 2) {
     	fprintf(stderr,"Usar: %s num_iteraciones\n", argv[0]);
     	exit(EXIT_FAILURE);
    }
	
    if (rank == 0) {
	    n = atol(argv[1]); //num. of intervals	
	}
	
    if (rank == 0) { 
		MPI_Win_create(&n, sizeof(int), 1, MPI_INFO_NULL,  MPI_COMM_WORLD, &nwin); 
		MPI_Win_create(&pi, sizeof(double), 1, MPI_INFO_NULL,  MPI_COMM_WORLD, &piwin);  
    } 
    else { 
		MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL,  MPI_COMM_WORLD, &nwin); 
		MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &piwin); 
    } 
	
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	
	MPI_Win_fence(0, nwin); 
	if (rank != 0) { 
	    MPI_Get(&n, 1, MPI_INT, 0, 0, 1, MPI_INT, nwin);
	}	
	MPI_Win_fence(0, nwin); 
	     
    h   = 1.0 / (double) n; 
    sum = 0.0; 
    for (i = rank; i < n; i += psize) { 
		x = h * ((double)i - 0.5); 
		sum += (4.0 / (1.0 + x*x)); 
    } 
    local_pi = h * sum; 
	
	MPI_Win_fence( 0, piwin); 
    MPI_Accumulate(&local_pi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, piwin); 
	MPI_Win_fence(0, piwin); 
	
	double end = MPI_Wtime();
		
    if (rank == 0) {
		printf("The obtained Pi value is: %.16f, the error is: %.16f\n", pi, fabs(pi - PI25DT));
		printf("elapsed time %f \n", end);
	} 
		
    MPI_Win_free(&nwin); 
    MPI_Win_free(&piwin); 
    MPI_Finalize(); 
	
    return 0; 
} 