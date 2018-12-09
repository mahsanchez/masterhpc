# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <time.h>


# define NX 2400
# define NY 2400

double u[NX][NY];
double udiff[NX][NY];
double uexact[NX][NY];
double unew[NX][NY];
double f[NX][NY];

double v;

int main ( int argc, char *argv[] );
double norm ( int nx, int ny, double a[NX][NY] );
void initial ( int nx, int ny, double f[NX][NY] );
void iteration ( int nx, int ny, double dx, double dy, double f[NX][NY],
double u[NX][NY], double unew[NX][NY] );
double evalexact ( double x, double y );
double evaldxdyexact ( double x, double y );


int main ( int argc, char *argv[] )
{
  int converged = 0;
  double diff;
  double dx;
  double dy;
  double error;
  //double f[NX][NY];
  int i;
  int it;
  int it_max = 1500;
  int j;
  int nx = NX;
  int ny = NY;
  double tolerance = 0.000001;
  //double u[NX][NY];
  double u_norm;
/*
  double udiff[NX][NY];
  double uexact[NX][NY];
  double unew[NX][NY];
  */
  double unew_norm;
  double x;
  double y;

  dx = 1.0 / ( double ) ( nx - 1 );
  dy = 1.0 / ( double ) ( ny - 1 );
/*
  Print a message.
*/
 
  printf ( "\n" );
    printf ( "  Number of X grid points: %d\n", nx );
    printf ( "  Number of Y grid points: %d\n", ny );
    printf ( "  X grid spacing: %f\n", dx );
    printf ( "  Y grid spacing: %f\n", dy );
	
clock_t tStart = clock();

/*
  Set the initial solution
*/
 #pragma omp parallel  
 {
	  /*
         Initialize the data
      */
      initial ( nx, ny, f );
      /*
         Set the initial solution
      */
	  #pragma omp for private(i, j)
	  for ( i = 0; i < nx; i++ )
	  {
		for ( j = 0; j < ny; j++ )
		{
		  if ( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
		  {
			unew[i][j] = f[i][j];
		  }
		  else
		  {
			unew[i][j] = 0.0;
		  }
		}
	  }	  	  
	  
      unew_norm = norm ( nx, ny, unew );	  
	  /*
      Set the exact solution.
      */	  
	  #pragma omp for private(i, j, x, y)
	  for ( i = 0; i < nx; i++ )
	  {
		x = ( double ) ( i ) / ( double ) ( nx - 1 );
		for ( j = 0; j < ny; j++ )
		{
		  y = ( double ) ( j ) / ( double ) ( ny - 1 );
		  uexact[i][j] = evalexact ( x, y );
		}
	  }
	  
      u_norm = norm ( nx, ny, uexact );
	  
	  #pragma omp single
      printf ( "  Norm of exact solution = %g\n", u_norm );	  
	  /*
       Iteration
      */	  
	  #pragma omp single
          printf ( " \n Step    ||Unew||     ||Diff||     ||Error||\n\n" );

      #pragma omp for private(i, j)
	  for ( i = 0; i < nx; i++ )
	  {
		for ( j = 0; j < ny; j++ )
		{
		  udiff[i][j] = unew[i][j] - uexact[i][j];
		}
	  }	 

	 error = norm ( nx, ny, udiff );
	 
	 #pragma omp single
	 printf ( "  %4d  %14g  %14g\n", 0, unew_norm, error );
	 	
  }
  
  
  for ( it = 1; it <= it_max; it++ ) 
  {
	#pragma omp parallel    
	{	
	    #pragma omp for private(i, j)
		for ( i = 0; i < nx; i++ )
		{
		  for ( j = 0; j < ny; j++ )
		  {
			u[i][j] = unew[i][j];
		  }
		}
		
		iteration ( nx, ny, dx, dy, f, u, unew );		
		/*
         Check for convergence.
        */ 
        #pragma omp single		
		u_norm = unew_norm; 
		    
        unew_norm = norm ( nx, ny, unew );

		#pragma omp for private(i, j)
		for ( i = 0; i < nx; i++ )
		{
		  for ( j = 0; j < ny; j++ )
		  {
			udiff[i][j] = unew[i][j] - u[i][j];
		  }
		}	
		 
        diff = norm ( nx, ny, udiff );
		
		#pragma omp for private(i, j)
		for ( i = 0; i < nx; i++ )
		{	
		  for ( j = 0; j < ny; j++ )
		  {
			udiff[i][j] = unew[i][j] - uexact[i][j];
		  }
		}
		
		error = norm ( nx, ny, udiff );
		
		#pragma omp single
		printf ( "  %4d  %14g  %14g  %14g\n", it, unew_norm, diff, error );
	}

	if ( diff <= tolerance )
	{
	  converged = 1;
	  break;
	}
  }

  if ( converged )
  {
    printf ( "  Converged\n" );
  }
  else
  {
    printf ( "  NOT converged.\n" );
  }
  
   printf("Time taken by Host: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
   
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );


  return 0;
}

double norm ( int nx, int ny, double a[NX][NY] )
{
  int i;
  int j;

  v = 0.0;

  #pragma omp for private(i, j) reduction(+:v)
  for ( i = 0; i < nx; i++ )
  {
    for ( j = 0; j < ny; j++ )
    {	
      v = v + a[i][j] * a[i][j];
    }
  }
  
  #pragma omp single
  v = sqrt ( v / ( double ) ( nx * ny )  );

  return v;
}


/******************************************************************************/

void initial ( int nx, int ny, double f[NX][NY] )
{
  double fnorm;
  int i;
  int j;
  double x;
  double y;

  #pragma omp for private(i, j, x, y)
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
  
  if (omp_get_thread_num() == 0) {
	  printf ( "  Norm of F = %g\n", fnorm );
  }

  return;
}

void iteration ( int nx, int ny, double dx, double dy, double f[NX][NY], double u[NX][NY], double unew[NX][NY] )
{
  int i;
  int j;

  #pragma omp for private(i, j)
  for ( i = 0; i < nx; i++ )
  {
	for ( j = 0; j < nx; j++ )
	{
	  if ( i == 0 || j == 0 || i == nx - 1 || j == ny - 1 )
	  {
		unew[i][j] = f[i][j];
	  }
	  else
	  { 
		unew[i][j] = 0.25 * ( 
		  u[i-1][j] + u[i][j+1] + u[i][j-1] + u[i+1][j] + f[i][j] * dx * dy );
	  }
	}
  }
	  
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
# undef NX
# undef NY

