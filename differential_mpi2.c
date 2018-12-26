#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define COLS  12
#define ROWS  8

# define NX 2400
# define NY 2400

MPI_Comm grid_comm;
int converged;
double diff;
double dx;
double dy;
double error;
double f[NX][NY];
int i;
int it;
int it_max = 1500;
int j;
int nx = NX;
int ny = NY;
double tolerance = 0.000001;
double u[NX][NY];
double u_norm;
double udiff[NX][NY];
double uexact[NX][NY];
double unew[NX][NY];
double unew_norm;
double x;
double y;

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		
    int ndims = 2;
    int dims[2] = {0, 0};
    int periods[2]= {0, 0};
	int reorder = 0;
	
	MPI_Dims_create(p, ndims, dims);
	printf("Dimensions X=%d Y=%d\n", dims[0], dims[1]);
	
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);

    double a[ROWS*COLS];
    const int NPROWS=dims[1];  /* number of rows in _decomposition_ 2 */
    const int NPCOLS=dims[0];  /* number of cols in _decomposition_ 3 */
    const int BLOCKROWS = ROWS/NPROWS;  /* number of rows in _block_ NPROWS */
    const int BLOCKCOLS = COLS/NPCOLS; /* number of cols in _block_  NPCOLS */

	dx = 1.0 / ( double ) ( nx - 1 );
    dy = 1.0 / ( double ) ( ny - 1 );
	
	/*
    Initialize the data
    */
    initial ( nx, ny, f );
	fnorm = norm ( nx, ny, f );
    if (rank == 0) { 
       printf ( "  Norm of F = %g\n", fnorm );
	}
	
    if (p != NPROWS*NPCOLS) {
        fprintf(stderr,"Error: number of PEs %d != %d x %d\n", p, NPROWS, NPCOLS);
        MPI_Finalize();
        exit(-1);
    }
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

    MPI_Scatterv(a, counts, disps, blocktype, b, BLOCKROWS*BLOCKCOLS, MPI_DOUBLE, 0, grid_comm);
	
    /* each proc prints it's "b" out, in order */
    for (int proc=0; proc<p; proc++) {
        if (proc == rank) {
            printf("Rank = %d\n", rank);
            if (rank == 0) {
                printf("Global matrix: \n");
                for (int ii=0; ii<ROWS; ii++) {
                    for (int jj=0; jj<COLS; jj++) {
                        printf("%3lf ",(double)a[ii*COLS+jj]);
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


    MPI_Finalize();

    return 0;
}

//MPI_PROC_NULL

double norm ( int nx, int ny, double a[NX][NY] )
{
  int i;
  int j;
  double v, gv;

  v = 0.0;

  for ( i = 0; i < nx; i++ )
  {
    for ( j = 0; j < ny; j++ )
    {
      v = v + a[i][j] * a[i][j];
    }
  }
  
  /* Reduce all of the local sums into the global sum */
  MPI_Allreduce(&v, &gv, 1, MPI_FLOAT, MPI_SUM, grid_comm);
  
  gv = sqrt ( gv / ( double ) ( nx * ny )  ); 

  return gv;
}


void initial ( int nx, int ny, double f[NX][NY] )
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
      if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
      {
        f[i][j] = evalexact ( x, y );
      }
      else
      {
        f[i][j] = - evaldxdyexact ( x, y );
      }
    }
  }

  fnorm = norm ( nx, ny, f );

  printf ( "  Norm of F = %g\n", fnorm );

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