#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <math.h>

// row-major order
#define ind(i,j) (j)*(bx+2)+(i)

int main(int argc, char **argv) 
{
  int rank,p;
  int n, energy, niters;
  int parameters[3]; 
  MPI_Request request;
  MPI_Status status;
 
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  if(argc < 4) {
	  if(rank == 0) printf("usage: stencil_mpi <n> <energy> <niters>\n");
	  exit(1);
   }

  if (rank==0) {
      parameters[0] = atoi(argv[1]);
	  parameters[1] = atoi(argv[2]);
	  parameters[2] = atoi(argv[3]);
  }
  
  MPI_Bcast(parameters, 3, MPI_INT, 0, MPI_COMM_WORLD);
  
  n = parameters[0]; 
  energy = parameters[1]; 
  niters = parameters[2];
  
  int pdims[2]={0,0};
  MPI_Dims_create(p, 2, pdims);
  int px = pdims[0];
  int py = pdims[1];
  
  // Find north, south, east and west neighbors
  int periods[2] = {0,0};
  MPI_Comm cartesian_topology;
  MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 0, &cartesian_topology);

  // Get x,y coordinates per process
  int coordinates[2];
  MPI_Cart_coords(cartesian_topology, rank, 2, coordinates);
  int rx = coordinates[0];
  int ry = coordinates[1];
   
  // Find neighbors ranks
  int source, north, south, east, west;
  MPI_Cart_shift(cartesian_topology, 0, 1, &west, &east);
  MPI_Cart_shift(cartesian_topology, 1, 1, &north, &south);

  // decompose the domain
  int bx = n/px; // block size in x
  int by = n/py; // block size in y
  int offx = rx*bx; // offset in x
  int offy = ry*by; // offset in y

  double *aold = (double*)calloc(1,(bx+2)*(by+2)*sizeof(double)); 
  double *anew = (double*)calloc(1,(bx+2)*(by+2)*sizeof(double)); 
  double *tmp;

  #define nsources 3
  int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};
  int locnsources=0; 
  int locsources[nsources][2]; 
  for (int i=0; i<nsources; ++i) { 
    int locx = sources[i][0] - offx;
    int locy = sources[i][1] - offy;
    if(locx >= 0 && locx < bx && locy >= 0 && locy < by) {
      locsources[locnsources][0] = locx+1; 
      locsources[locnsources][1] = locy+1; 
      locnsources++;
    }
  }
  
  double t=-MPI_Wtime(); 
  
  double heat; // total heat in system
  for(int iter=0; iter<niters; ++iter) {
    // refresh heat sources
    for(int i=0; i<locnsources; ++i) {
      aold[ind(locsources[i][0],locsources[i][1])] += energy; // heat source
    }

    // exchange data with neighbors
    double *sendbuf = (double*)calloc(1,2*bx*sizeof(double)+2*by*sizeof(double)); 
    double *recvbuf = (double*)calloc(1,2*bx*sizeof(double)+2*by*sizeof(double)); 
	
    for(int i=0; i<by; ++i) sendbuf[i] = aold[ind(1,i+1)]; 
    for(int i=0; i<by; ++i) sendbuf[by+i] = aold[ind(bx,i+1)]; 
    for(int i=0; i<bx; ++i) sendbuf[2*by+i] = aold[ind(i+1,1)]; 
    for(int i=0; i<bx; ++i) sendbuf[2*by+bx+i] = aold[ind(i+1,by)]; 
	
    int counts[4] = {by, by, bx, bx};
    int displs[4] = {0, by, 2*by, 2*by+bx};
    MPI_Ineighbor_alltoallv(sendbuf, counts, displs, MPI_DOUBLE, recvbuf, counts, displs, MPI_DOUBLE, cartesian_topology, &request);
    MPI_Wait(&request, &status);
	
    for(int i=0; i<by; ++i) aold[ind(0,i+1)] = recvbuf[i];
    for(int i=0; i<by; ++i) aold[ind(bx+1,i+1)] = recvbuf[by+i]; 
    for(int i=0; i<bx; ++i) aold[ind(i+1,0)] = recvbuf[2*by+i]; 
    for(int i=0; i<bx; ++i) aold[ind(i+1,by+1)] = recvbuf[2*by+bx+i]; 

    heat = 0.0;
    for(int j=1; j<by+1; ++j) {
      for(int i=1; i<bx+1; ++i) {
        anew[ind(i,j)] = aold[ind(i,j)]/2.0 + (aold[ind(i-1,j)] + aold[ind(i+1,j)] + aold[ind(i,j-1)] + aold[ind(i,j+1)])/4.0/2.0;
        heat += anew[ind(i,j)];
      }
    }

    tmp=anew; anew=aold; aold=tmp; // swap arrays
  }
  
  
  // get final heat in the system
  double global_heat;
  MPI_Reduce(&heat, &global_heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
   t+=MPI_Wtime();
  
  if (rank == 0) {
	  printf("[%i] last heat: %f time: %f\n", rank, global_heat, t);
  }

  MPI_Finalize();
}
