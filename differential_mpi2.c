#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define COLS  12
#define ROWS  8

# define NX 12
# define NY 8

MPI_Comm grid_comm;
int converged;
double diff;
double dx;
double dy;
double error;
double f[NX*NY];
int i;
int it;
int it_max = 1500;
int j;
int nx = NX;
int ny = NY;
double tolerance = 0.000001;
double u[NX*NY];
double u_halo[NX][NY];
double u_norm;
double udiff[NX*NY];
double uexact[NX*NY];
double unew[NX*NY];
double unew_norm;
double x;
double y;
int p, rank;
double *halo_send;
double *halo_recv;

int main ( int argc, char *argv[] );
double norm ( int nx, int ny, double *a );
void initial ( int nx, int ny, double *f );
void iteration ( int nx, int ny, double dx, double dy, double f[NX*NY], double u[NX*NY], double unew[NX*NY] );
double evalexact ( double x, double y );
double evaldxdyexact ( double x, double y );

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		
    int ndims = 2;
    int dims[2] = {0, 0};
    int periods[2]= {0, 0};
	int reorder = 0;
	
	MPI_Dims_create(p, ndims, dims);
	if (rank == 0) printf("Dimensions X=%d Y=%d\n", dims[0], dims[1]);
	
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);

    const int NPROWS=dims[1];  /* number of rows in _decomposition_ 2 */
    const int NPCOLS=dims[0];  /* number of cols in _decomposition_ 3 */
    const int BLOCKROWS = NY/NPROWS;  /* number of rows in _block_ NPROWS */
    const int BLOCKCOLS = NX/NPCOLS; /* number of cols in _block_  NPCOLS */
	
    if (p != NPROWS*NPCOLS) {
        fprintf(stderr,"Error: number of PEs %d != %d x %d\n", p, NPROWS, NPCOLS);
        MPI_Finalize();
        exit(-1);
    }

	dx = 1.0 / ( double ) ( nx - 1 );
    dy = 1.0 / ( double ) ( ny - 1 );
	
	/*
    Print a message.
    */ 
	if (rank == 0) {
		printf ( "\n" );
		printf ( "  Number of X grid points: %d\n", NX );
		printf ( "  Number of Y grid points: %d\n", NY );
		printf ( "  X grid spacing: %f\n", dx );
		printf ( "  Y grid spacing: %f\n", dy );
	}
	
	/*
    Initialize the data
    */
	if (rank == 0) {
       initial ( NX, NY, f );
	}
	
	MPI_Barrier(grid_comm);
	
	/*
	Distribute the matrix in blocks across all process checkerboard distribution
	*/
    double b[BLOCKROWS*BLOCKCOLS];
    for (int ii=0; ii<BLOCKROWS*BLOCKCOLS; ii++) b[ii] = 0;

    MPI_Datatype blocktype;
    MPI_Datatype blocktype2;

    MPI_Type_vector(BLOCKROWS, BLOCKCOLS, COLS, MPI_DOUBLE, &blocktype2);
    MPI_Type_create_resized( blocktype2, 0, sizeof(double), &blocktype);
    MPI_Type_commit(&blocktype);

    int disps[NPROWS*NPCOLS];
    int counts[NPROWS*NPCOLS];
    for (int ii=0; ii<NPROWS; ii++) {
        for (int jj=0; jj<NPCOLS; jj++) {
            disps[ii*NPCOLS+jj] = ii*COLS*BLOCKROWS+jj*BLOCKCOLS;
            counts [ii*NPCOLS+jj] = 1;
        }
    }

    MPI_Scatterv(f, counts, disps, blocktype, b, BLOCKROWS*BLOCKCOLS, MPI_DOUBLE, 0, grid_comm);
	
	// Update ny, nx to local sub matrix block size per process
	ny = BLOCKROWS; 
    nx = BLOCKCOLS;
	//	
	double fnorm = norm (nx, ny, b );
    if (rank == 0) { 
      printf ( "  Norm of F = %g\n", fnorm );
	}
		
	/*
      Set the initial solution
   */
   for ( i = 0; i < nx; i++ )
   {
	for ( j = 0; j < ny; j++ )
	{
	  if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
	  {
		unew[i+j*nx] = f[i+j*nx];
	  }
	  else
	  {
		unew[i+j*nx] = 0.0;
	  }
	}
   } 
   unew_norm = norm ( nx, ny, unew );
   if (rank == 0) { 
	  printf ( "  Norm of unew_norm = %g\n", unew_norm );
   }
	
  /*
  Set the exact solution.
   */
  for ( i = 0; i < nx; i++ )
  {
    x = ( double ) ( i ) / ( double ) ( NX - 1 );
    for ( j = 0; j < ny; j++ )
    {
      y = ( double ) ( j ) / ( double ) ( NY - 1 );
      uexact[i+j*nx] = evalexact ( x, y );
    }
  }
  u_norm = norm ( nx, ny, uexact );
  if (rank == 0) {
     printf ( "  Norm of exact solution = %g\n", u_norm );
  }
  
  /*
  Iteration
  */
  converged = 0;

  if (rank == 0) {
	  printf ( "\n" );
	  printf ( "  Step    ||Unew||     ||Diff||     ||Error||\n" );
	  printf ( "\n" );
  }

  for ( i = 0; i < nx; i++ )
  {
    for ( j = 0; j < ny; j++ )
    {
      udiff[i+j*nx] = unew[i+j*nx] - uexact[i+j*nx];
    }
  }
  error = norm ( nx, ny, udiff );
  if (rank == 0) {
	printf ( "  %4d  %14g                  %14g\n", 0, unew_norm, error );
  }

  for ( it = 1; it <= it_max; it++ )
  {
    for ( i = 0; i < nx; i++ )
    {
      for ( j = 0; j < ny; j++ )
      { 
		// for u iterate index from 1..ny and 1..nx then store halo data in the borders 
        u[i+j*nx] = unew[i+j*nx];
      }
    }
	
	// exchange halo between neighbors 
	halo_exchange(nx, ny);

	// apply jacobi
    iteration ( nx, ny, dx, dy, b, u, unew );

    // Check for convergence.
    u_norm = unew_norm;
    unew_norm = norm ( nx, ny, unew );

    for ( i = 0; i < nx; i++ )
    {
      for ( j = 0; j < ny; j++ )
      {
        udiff[i+j*nx] = unew[i+j*nx] - u[i+j*nx];
      }
    }
    diff = norm ( nx, ny, udiff );

    for ( i = 0; i < nx; i++ )
    {
      for ( j = 0; j < ny; j++ )
      {
        udiff[i+j*nx] = unew[i+j*nx] - uexact[i+j*nx];
      }
    }
    error = norm ( nx, ny, udiff );

	if (rank == 0) {
		printf ( "  %4d  %14g  %14g  %14g\n", it, unew_norm, diff, error );
	}

    if ( diff <= tolerance )
    {
      converged = 1;
      break;
    }

  }

  if (rank == 0) {
	  if ( converged )
	  {
		printf ( "  Converged\n" );
	  }
	  else
	  {
		printf ( "  NOT converged.\n" );
	  }	  
	  printf ( "\n" );
      printf ( "  Normal end of execution.\n" );
      printf ( "\n" );
  }

/*
  Terminate.
*/
	
    /* each proc prints it's "b" out, in order */
	/*
    for (int proc=0; proc<p; proc++) {
        if (proc == rank) {
            printf("Rank = %d\n", rank);
            if (rank == 0) {
                printf("Global matrix: \n");
                for (int ii=0; ii<ROWS; ii++) {
                    for (int jj=0; jj<COLS; jj++) {
                        printf("%3lf ",(double)f[ii*COLS+jj]);
                    }
                    printf("\n");
                }
            }
            printf("Local Matrix:\n");
            for (int ii=0; ii<BLOCKROWS; ii++) {
                for (int jj=0; jj<BLOCKCOLS; jj++) {
                    printf("%3lf ",(double)b[ii*BLOCKCOLS+jj]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(grid_comm);
    }
   */

    MPI_Finalize();

    return 0;
}


double norm (int nx, int ny, double *a)
{
  int i;
  int j;
  double v, gv;

  v = 0.0;

  for ( i = 0; i < nx; i++ )
  {
    for ( j = 0; j < ny; j++ )
    {
      v = v + a[i+j*nx] * a[i+j*nx];
    }
  }
  
  /* Reduce all of the local sums into the global sum */
  MPI_Allreduce(&v, &gv, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
  
  gv = sqrt ( gv / ( double ) (NX * NY)  ); 

  return gv;
}

MPI_Status status;
MPI_Request request;

void halo_exchange(int nx, int ny) {
   // halo exchange direction 0
  int direction = 0;
  int rank_source;
  int rank_dest;
  int disp = 1;
  
  MPI_Cart_shift(grid_comm, direction, disp, &rank_source, &rank_dest);
  
  // copy left corner halo
  if (rank_source != MPI_PROC_NULL) {
	 for (int j = 0; j < ny; j++) 
		 halo_buf[i] = u[1][j];
	 
	 MPI_Sendrecv(halo_send, nx, MPI_DOUBLE, rank_source, halo_recv, tag, grid_comm);

	 for (int j = 0; j < ny; i++) 
		 u[0][j+1] = halo_recv[j];
  }
  
  // copy right corner halo
  if (rank_source != MPI_PROC_NULL) {
	 for (int j = 0; j < ny; j++) 
		 halo_buf[i] = u[nx-1][j];
	 
	 MPI_Sendrecv(halo_send, nx, MPI_DOUBLE, rank_source, halo_recv, tag, grid_comm);

	 for (int j = 0; j < ny; i++) 
		 u[ny][j+1] = halo_recv[j];
  }
  
  MPI_Cart_shift(grid_comm, ++direction, disp, &rank_source, &rank_dest);
  
  // top halo
  if (rank_source != MPI_PROC_NULL) {
	 for (int j = 0; j < ny; j++) 
		 halo_buf[i] = u[j][1];
	 
	 MPI_Sendrecv(halo_send, nx, MPI_DOUBLE, rank_source, &u[0][1], nx, tag, grid_comm);
  }
  
  // botton halo
  if (rank_source != MPI_PROC_NULL) {
	 for (int j = 0; j < ny; j++) 
		 halo_buf[i] = u[j][ny-1];
	 
	 MPI_Sendrecv(halo_send, nx, MPI_DOUBLE, rank_source, &u[ny][1], tag, grid_comm);
  }
  
}

void iteration ( int nx, int ny, double dx, double dy, double *f, double *u, double *unew )
{
  int i;
  int j;
  int rank_source, rank_dest;
  int disp = 1;
  int boundary = 0;
  int tag = 0;
  
  // Apply Stencil
 
  for ( i = 0; i < nx; i++ )
  {
    for ( j = 0; j < nx; j++ )
    {
	  // i == 0 || j == 0 || i == nx - 1 || j == ny - 1 
      if ( boundary )
      {
        unew[i+j*nx] = f[i+j*nx];
      }
      else
      { 
        unew[i + j*nx] = 0.25 * ( 
          u[(i-1)][j] + u[i][j+1] + u_halo[i][j-1] + u_halo[i][j] + f[i + j*nx] * dx * dy );
      }
    }
  }
  return;
}


void initial ( int nx, int ny, double f[NX*NY]  )
{
  double fnorm;
  int i;
  int j;
  double x;
  double y;

  for ( i = 0; i < nx; i++ )
  {
    x = ( double ) ( i ) / ( double ) ( nx - 1 );
    for ( j = 0; j < ny; j++ )
    {
      y = ( double ) ( j ) / ( double ) ( ny - 1 );
      if ( i == 0 || i == nx - 1 || j == 0 || j*nx == ny*nx - 1 )
      {
        f[i+j*nx] = evalexact ( x, y );
      }
      else
      {
        f[i+j*nx] = - evaldxdyexact ( x, y );
      }
    }
  }

  //fnorm = norm ( nx, ny, f );
  //printf ( "  Norm of F = %g\n", fnorm );

  return;
}

double evalexact ( double x, double y )
{
  double pi = 3.141592653589793;
  double value;

  value = sin ( pi * x * y );

  return value;
}


double evaldxdyexact ( double x, double y )
{
  double pi = 3.141592653589793;
  double value;

  value = - pi * pi * ( x * x + y * y ) * sin ( pi * x * y );

  return value;
}