 /* Matrix-vector product. Matrix of N*M */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

float A[8192][8192];
float b[1024][1024];
float by[1024];

int main(int argc,char *argv[])
{
    int i, j, N, M;
	int myrank, P;
	int provided;
    //float **A, *Avector, *x, *y, temp;
	float *x, *y, temp;
	

    if (argc < 3) {
     	fprintf(stderr,"Usar: %s filas columnas\n", argv[0]);
     	exit(EXIT_FAILURE);
    }

    N = atoi(argv[1]);  //Rows of the matrix and elements of vector y
    M = atoi(argv[2]);  //Columns of the matrix and elements of vector x
	
	//MPI_Init (&argc, &argv);
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    printf("Nivel proporcionado %d de %d, %d, %d, %d\n", provided, MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
	
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);	/* who am i */
    MPI_Comm_size(MPI_COMM_WORLD, &P); /* number of processors */
	
	if (N%P !=0) {
       if (myrank==0) printf("Vector size not divisible by number of processors\n");
       MPI_Finalize();
       exit(-1);
    }

	/*
   if((Avector = (float *) malloc(N*M*sizeof(float))) == NULL)
        printf("Error en malloc Avector[%d]\n",N*M);

   if((A = (float **) malloc(N*sizeof(float *))) == NULL)
        printf("Error en malloc del array de %d punteros\n",N);
	*/
		  
   //for (i=0;i<N;i++)
	//   A[i] = (float *)malloc(M * sizeof(float)); 
  	//*(A+i) = Avector+i*M;

   if (myrank == 0) {
	   for(i=0; i<N; i++)
		for(j=0; j<M; j++) {
			A[i][j] = (0.15*i - 0.1*j)/N;
		}	
   }
   
    if((x = (float *) malloc(M*sizeof(float))) == NULL)
        printf("Error en malloc x[%d]\n",M);
	
    if((y = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error en malloc y[%d]\n",N);

	if (myrank == 0) {
		for(i=0; i<M; i++)
			 x[i] = (M/2.0 - i);
	}
	
	MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD); 
    MPI_Scatter(A, M * (N/P), MPI_FLOAT, b, M * (N/P), MPI_FLOAT, 0, MPI_COMM_WORLD);

    #pragma omp parallel for default(none) private(i, j, temp) shared(N, M, P, A, x, by) 
    for(i=0; i < N/P; i++){
	   temp = 0.0;
	   for(j=0; j < M; j++) {
			temp += A[i][j] * x[j];
	   }
	   by[i] = temp;
    }
	
	MPI_Gather (by, N/P, MPI_FLOAT, y, N/P, MPI_FLOAT, 0, MPI_COMM_WORLD);
	    
	if (myrank == 0) {
	   printf("Done,  y[0] = %g  y[%d] = %g \n", y[0], N-1, y[N-1]);
	}
	
	MPI_Finalize();
    

    return 0;
}



