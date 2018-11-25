#include <stdio.h>
#include <math.h>
#include "mpi.h"

/*
mpiexec -n 4 canon-mpi 16
mpiexec -n 16 canon-mpi 256
mpiexec -n 16 canon-mpi 512
*/

void print_matrix(const char* desc, int m, int n, float* a, int lda) {
	int i, j;
	printf("\n %s\n", desc);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf(" %4.7f", a[i*lda + j]);
		}	
		printf("\n");
	}
}

void print_intmatrix(const char* desc, int m, int n, int* a, int lda) {
	int i, j;
	printf("\n %s\n", desc);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf(" %d", a[i*lda + j]);
		}	
		printf("\n");
	}
}

void mmult(float *tileA, float *tileB, float *tileC, int TILE_WIDTH) {
	for (int i = 0; i < TILE_WIDTH; i++) {
		for (int j = 0; j < TILE_WIDTH; j++) {
			for (int k = 0; k < TILE_WIDTH; k++) {
				tileC[i*TILE_WIDTH + j] += tileA[i*TILE_WIDTH + k] * tileB[j +k*TILE_WIDTH];
			}
		}
	}
}

int main ( int argc , char **argv )
{
     int rank;
	 int value, nproc , false=0;
     MPI_Status status, status2;
	 float *A, *B, *C;
	 
	int N = argc > 0 ? atoi(argv[1]) : 16;
	int rows = sqrt(N);
	
	 MPI_Request request;
	 MPI_Request request2, request3, request4;
	 
	int dims[2], periods[2], coords[2];
	int ndims = 2;
    int right, left, up, down;
	int reorder = 0;
	dims[0] = (int) sqrt(rows); dims[1] = (int) sqrt(rows);
	periods[0] = 1; periods[1] = 1;
	int cart_rank, cart_size;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	
	//MPI_Dims_create(nproc, 2, dims);
	//print_intmatrix("dims: ", 1, 2, dims, 2);
	//print_intmatrix("periods: ", 1, 2, periods, 2);
	
    MPI_Comm comm_cart;
	MPI_Dims_create(nproc, ndims, dims);
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
    MPI_Comm_rank(comm_cart, &cart_rank );
    MPI_Comm_size(comm_cart, &cart_size );
	MPI_Cart_shift(comm_cart, 1, 1, &left, &right);
    MPI_Cart_shift(comm_cart, 0, 1, &up, &down);
	 
	 if (rank == 0) {
		A = (float *) malloc(N * sizeof(float)); 
		B = (float *) malloc(N * sizeof(float)); 
		C = (float *) malloc(N * sizeof(float)); 
		/* Matrix inicialization */
        for (int i=0; i < rows; i++) {
			for(int j=0; j < rows; j++) {
				A[i*rows + j] = (0.15*i - 0.1*j)/N;
		     	B[i*rows + j] = (0.15*i - 0.1*j)/N;
			}	
	    }		
		print_matrix("A: ", rows, rows, A, rows);
     }
	 
	 // generate submatrix for each process and scatter it among them
	 int TILE_SIZE = (rows * rows)/ nproc; 
	 int TILE_WIDTH = (int) sqrt(TILE_SIZE);
	 float *tileA = (float *) malloc(TILE_SIZE * sizeof(float)); 
	 float *tileB = (float *) malloc(TILE_SIZE * sizeof(float)); 
	 
     MPI_Scatter(A, TILE_SIZE, MPI_FLOAT, tileA, TILE_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
	 MPI_Scatter(B, TILE_SIZE, MPI_FLOAT, tileB, TILE_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
	 
	 // Canon algorithm using cartesian
	float *tileC = (float *) malloc(TILE_SIZE * sizeof(float)); 
	memset(tileC, 0, TILE_WIDTH);
	
	double time_st, time_end, elapsed;
	double maxtime,mintime,avgtime;
	time_st = MPI_Wtime();
	
	for (int i = 0; i < TILE_WIDTH; i++) {
		mmult(tileA, tileB, tileC, TILE_WIDTH);	
		MPI_Isend(tileA, TILE_WIDTH, MPI_FLOAT, left, 0, comm_cart, &request);
		MPI_Isend(tileB, TILE_WIDTH, MPI_FLOAT, up, 0, comm_cart, &request2);
		MPI_Recv(tileA, TILE_WIDTH, MPI_FLOAT, right, 0, comm_cart,  &request3);
		MPI_Recv(tileB, TILE_WIDTH, MPI_FLOAT, down, 0, comm_cart,  &request4);
		MPI_Wait(&request, &status);
		MPI_Wait(&request2, &status2);
	}
	
	time_end = MPI_Wtime();
    elapsed = time_end-time_st;
	
	//printf( "Rank: %d Cart Rank: %d got Left: %d Right: %d Up: %d  Down: %d\n", rank , cart_rank, left, right,  up, down);
	//print_matrix("TILE A: ", TILE_WIDTH, TILE_WIDTH, tileA, TILE_WIDTH);
	//print_matrix("TILE B: ", TILE_WIDTH, TILE_WIDTH, tileB, TILE_WIDTH);
	//print_matrix("TILE C: ", TILE_WIDTH, TILE_WIDTH, tileC, TILE_WIDTH);
	
	MPI_Gather(tileC, TILE_SIZE, MPI_FLOAT, C, TILE_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	/*compute max, min, and average timing statistics*/
    MPI_Reduce(&elapsed, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if (rank == 0) {
      avgtime = avgtime/nproc;
	  print_matrix("MATRIX C: ", rows, rows, C, rows);
      printf("Runtime Statistics: Min: %lf  Max: %lf  Avg:  %lf\n",mintime, maxtime,avgtime);
    }

    MPI_Finalize();
	
    return 0;
}