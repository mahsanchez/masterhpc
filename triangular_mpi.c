#include "mpi.h"
#include <stdio.h>

#define SIZE 16
#define ROW_SIZE 4

int main(int argc, char *argv[])
{
    int rank, nproc;
    int blocklen[ROW_SIZE];
    int displacement[ROW_SIZE];
    float matrix[SIZE];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if (nproc < 2)
    {
        printf("Please run with 2 processes.\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	for (int i = 0; i < ROW_SIZE; i++) {
		displacement[i] = ROW_SIZE*i;
		blocklen[i] = i + 1;
	}
	
	/*
	for (int i=0; i<ROW_SIZE; i++)
            printf("displacement[%d] = %d\n", i, displacement[i]);
		
	for (int i=0; i<ROW_SIZE; i++)
            printf("blocklen[%d] = %d\n", i, blocklen[i]);
	*/
	
	MPI_Datatype type;
	MPI_Datatype type2;
	
	MPI_Type_contiguous(ROW_SIZE, MPI_FLOAT, &type2);
    MPI_Type_commit(&type2);
    MPI_Type_indexed(ROW_SIZE, blocklen, displacement, MPI_FLOAT, &type);
    MPI_Type_commit(&type);	
	
	int dst = 1;
	int src = 0;
	int tag = 123;

    if (rank == 0)
    {
        for (int i=0; i<SIZE; i++) {
            matrix[i] = i;
		}	
		
        MPI_Send(matrix, 1, type, dst, tag, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        for (int i=0; i<SIZE; i++)
            matrix[i] = 0;
			
        MPI_Recv(matrix, 1, type, src, tag, MPI_COMM_WORLD, &status);
		
        for (int i=0; i<SIZE; i++)
            printf("matrix[%d] = %f\n", i, matrix[i]);
    }
	

    MPI_Finalize();
    return 0;
}