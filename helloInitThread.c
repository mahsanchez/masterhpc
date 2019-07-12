#include <stdio.h>
#include "mpi.h"
#include <omp.h>

int main(int argc, char *argv[]) {
  int myid, npes, iam, nt, namelen, proporcionado;
  char node_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &proporcionado);
  printf("Nivel proporcionado %d de %d, %d, %d, %d\n", proporcionado, MPI_THREAD_SINGLE, 
		 MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(node_name, &namelen);

  #pragma omp parallel default(shared) private(iam, nt)
  {
    nt = omp_get_num_threads();
    iam = omp_get_thread_num();
    printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
           iam, nt, myid, npes, node_name);
  }

  MPI_Finalize();
}

