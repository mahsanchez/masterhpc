#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <math.h>

// row-major order
#define ind(i,j) (j)*(bx+2)+(i)

int main(int argc, char **argv) {
  int rank, p;
  int parameters[3];
  int n, energy, niters;
  
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

  // allocate two work arrays
  double *aold = (double*)calloc(1,(bx+2)*(by+2)*sizeof(double)); 
  double *anew = (double*)calloc(1,(bx+2)*(by+2)*sizeof(double)); 
  double *tmp;

  // initialize three heat sources
  #define nsources 3
  int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};
  int locnsources=0; // number of sources in my area
  int locsources[nsources][2]; // sources local to my rank
  for (int i=0; i<nsources; ++i) { // determine which sources are in my patch
    int locx = sources[i][0] - offx;
    int locy = sources[i][1] - offy;
    if(locx >= 0 && locx < bx && locy >= 0 && locy < by) {
      locsources[locnsources][0] = locx+1; // offset by halo zone
      locsources[locnsources][1] = locy+1; // offset by halo zone
      locnsources++;
    }
  }

  double t=-MPI_Wtime(); // take time
  // allocate communication buffers
  double *sbufnorth = (double*)calloc(1,bx*sizeof(double)); // send buffers
  double *sbufsouth = (double*)calloc(1,bx*sizeof(double));
  double *sbufeast = (double*)calloc(1,by*sizeof(double));
  double *sbufwest = (double*)calloc(1,by*sizeof(double));
  
  double *rbufnorth = (double*)calloc(1,bx*sizeof(double)); // receive buffers
  double *rbufsouth = (double*)calloc(1,bx*sizeof(double));
  double *rbufeast = (double*)calloc(1,by*sizeof(double));
  double *rbufwest = (double*)calloc(1,by*sizeof(double));

  double heat; // total heat in system
  for(int iter=0; iter<niters; ++iter) {
    // refresh heat sources
    for(int i=0; i<locnsources; ++i) {
      aold[ind(locsources[i][0],locsources[i][1])] += energy; // heat source
    }

    // exchange data with neighbors
    MPI_Request reqs[8];
    for(int i=0; i<bx; ++i) sbufnorth[i] = aold[ind(i+1,1)]; // pack loop - last valid region
    for(int i=0; i<bx; ++i) sbufsouth[i] = aold[ind(i+1,by)]; // pack loop
    for(int i=0; i<by; ++i) sbufeast[i] = aold[ind(bx,i+1)]; // pack loop
    for(int i=0; i<by; ++i) sbufwest[i] = aold[ind(1,i+1)]; // pack loop
	
    MPI_Isend(sbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(sbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(sbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(sbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[3]);
    MPI_Irecv(rbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[4]);
    MPI_Irecv(rbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[5]);
    MPI_Irecv(rbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[6]);
    MPI_Irecv(rbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[7]);
    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
	
    for(int i=0; i<bx; ++i) aold[ind(i+1,0)] = rbufnorth[i]; // unpack loop - into ghost cells
    for(int i=0; i<bx; ++i) aold[ind(i+1,by+1)] = rbufsouth[i]; // unpack loop
    for(int i=0; i<by; ++i) aold[ind(bx+1,i+1)] = rbufeast[i]; // unpack loop
    for(int i=0; i<by; ++i) aold[ind(0,i+1)] = rbufwest[i]; // unpack loop

    // update grid points
    heat = 0.0;
    for(int j=1; j<by+1; ++j) {
      for(int i=1; i<bx+1; ++i) {
        anew[ind(i,j)] = aold[ind(i,j)]/2.0 + (aold[ind(i-1,j)] + aold[ind(i+1,j)] + aold[ind(i,j-1)] + aold[ind(i,j+1)])/4.0/2.0;
        heat += anew[ind(i,j)];
      }
    }

    // swap arrays
    tmp=anew; anew=aold; aold=tmp;
  }
  t+=MPI_Wtime();

  // get final heat in the system
  double global_heat;
  MPI_Reduce(&heat, &global_heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(rank == 0) {
	  printf("[%i] last heat: %f time: %f\n", rank, global_heat, t);
  }

  MPI_Finalize();
}
