/* Obtain the PI value using the numerical integration of 4/(1+x*x) between 0 and 1.
The numerical integration is calculated with n rectangular intervals (area=(1/n)*4/(1+x*x))) and adding the area of all these rectangle intervals  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include <omp.h>

int main(argc,argv)
int argc;
char *argv[];
{
    long int n=1000000000, i;
    double PI25DT = 3.141592653589793238462643;
    double pi, local_pi, h, sum, x;
	int rank, psize, provided;
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    printf("Nivel proporcionado %d de %d, %d, %d, %d\n", provided, MPI_THREAD_SINGLE,  MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
	
    MPI_Comm_size(MPI_COMM_WORLD,&psize);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (argc < 2) {
     	fprintf(stderr,"Usar: %s num_iteraciones\n", argv[0]);
     	exit(EXIT_FAILURE);
    }
	
    if (rank == 0) {
	    n = atol(argv[1]); //num. of intervals	
	}
	
    MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
    h   = 1.0 / (double) n;  //wide of the rectangle
    sum = 0.0;
	
	# pragma omp parallel for reduction(+:sum) private(x)
    for (i = rank; i < n; i += psize) {
		x = h * ((double)i + 0.5);   //height of the rectangle
        sum += 4.0 / (1.0 + x*x);
    }
	
    local_pi = h * sum;
	
	MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	double end = MPI_Wtime();
    

	if (rank == 0) {
		printf("The obtained Pi value is: %.16f, the error is: %.16f\n", pi, fabs(pi - PI25DT));
		printf("elapsed time %f \n", end);
	}
    

	MPI_Finalize();
    return 0;
}
