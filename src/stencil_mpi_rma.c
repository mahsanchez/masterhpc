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
  double global_heat;
  int parameters[3];
  MPI_Win grid_win, heat_win;
  
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

  // Find neighbors ranks
  int source, north, south, east, west;
  MPI_Cart_shift(cartesian_topology, 0, 1, &west, &east);
  MPI_Cart_shift(cartesian_topology, 1, 1, &north, &south);
  
  // decompose the domain
  int bx = n/px; // tile size in x
  int by = n/py; // tile size in y
  int offsetx = rank * bx; // offset in x
  int offsety = rank * by; // offset in y

  int size = (bx+2)*(by+2); // process-local grid (including halos (thus +2))
  double *grid_mem = (double*)calloc(1,2*size*sizeof(double));
  MPI_Win_create(grid_mem, 2*size*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &grid_win);
  MPI_Win_create(&global_heat, sizeof(double), sizeof(double), MPI_INFO_NULL,  MPI_COMM_WORLD, &heat_win);

  double *tmp;
  double *anew=grid_mem; // each rank's offset
  double *aold=grid_mem + size; // second half is aold! 

  #define nsources 3
  int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};
  
  int locnsources=0; // number of sources in my area
  int locsources[nsources][2]; // sources local to my rank
  for (int i=0; i<nsources; ++i) { // determine which sources are in my patch
    int locx = sources[i][0] - offsetx;
    int locy = sources[i][1] - offsety;
    if(locx >= 0 && locx < bx && locy >= 0 && locy < by) {
      locsources[locnsources][0] = locx+1; // offset by halo zone
      locsources[locnsources][1] = locy+1; // offset by halo zone
      locnsources++;
    }
  }

  double t=-MPI_Wtime(); 
  // create north-south datatype
  MPI_Datatype north_south_type;
  MPI_Type_contiguous(bx, MPI_DOUBLE, &north_south_type);
  MPI_Type_commit(&north_south_type);
  
  // create east-west type
  MPI_Datatype east_west_type;
  MPI_Type_vector(by,1,bx+2,MPI_DOUBLE, &east_west_type);
  MPI_Type_commit(&east_west_type);

  double heat; // total heat in system
  for(int iter=0; iter<niters; ++iter) {
    // refresh heat sources
    for(int i=0; i<locnsources; ++i) {
      aold[ind(locsources[i][0],locsources[i][1])] += energy; // heat source
    }
	
	// exchange data with neighbors
	int offset = size*((iter+1)%2);
	
    MPI_Win_fence(0, grid_win);
    MPI_Put(&aold[ind(1,1)], 1, north_south_type, north, ind(1,by+1)+offset, 1, north_south_type, grid_win);
    MPI_Put(&aold[ind(1,by)], 1, north_south_type, south, ind(1,0)+offset, 1, north_south_type, grid_win);
    MPI_Put(&aold[ind(bx,1)], 1, east_west_type, east, ind(0,1)+offset, 1, east_west_type, grid_win);
    MPI_Put(&aold[ind(1,1)], 1, east_west_type, west, ind(bx+1,1)+offset, 1, east_west_type, grid_win);
    MPI_Win_fence(0, grid_win);

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
 

  //Gather all results and get final heat in the system
	MPI_Win_fence(0, heat_win);	
	MPI_Accumulate(&heat, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, heat_win);
	MPI_Win_fence(0, heat_win);	
  
   t+=MPI_Wtime();
  
  if(rank == 0) {
	  printf("[%i] last heat: %f time: %f\n", rank, global_heat, t);
  }	  
  
  MPI_Type_free(&east_west_type);
  MPI_Type_free(&north_south_type);
  
  MPI_Win_free(&grid_win);
  MPI_Win_free(&heat_win);

  MPI_Finalize();
}
