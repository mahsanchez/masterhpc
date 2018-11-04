#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl_lapacke.h"

/*
   The program computes the solution to the system of linear
   equations with a symmetric positive-definite matrix A and multiple
   right-hand sides B, where A is the coefficient matrix:
 
     3.14   0.17  -0.90   1.65  -0.72
     0.17   0.79   0.83  -0.65   0.28
    -0.90   0.83   4.53  -3.70   1.60
     1.65  -0.65  -3.70   5.32  -1.37
    -0.72   0.28   1.60  -1.37   1.98

   and B is the right-hand side matrix:
 
    -7.29   6.11   0.59
     9.25   2.90   8.88
     5.99  -5.05   7.57
    -1.94  -3.80   5.57
    -8.30   9.66  -1.67
 
 Example Program Results

 Solution
  -6.02   3.95  -3.14
  15.62   4.32  13.05
   3.02  -8.25   4.91
   3.25  -4.83   6.11
  -8.78   9.04  -3.57

 Details of Cholesky factorization
   1.77   0.10  -0.51   0.93  -0.41
   0.00   0.88   0.99  -0.84   0.36
   0.00   0.00   1.81  -1.32   0.57
   0.00   0.00   0.00   1.42   0.05
   0.00   0.00   0.00   0.00   1.16
*/

double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);

    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

double *generate_spd_matrix(int n, int *info) {
	int i;
    double *matrix = generate_matrix(n);
	double *spd_matrix = (double *)malloc(sizeof(double) * n * n);

    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			double sum = 0;
			for(int k=0; k<n; k++) {
				sum += matrix[i*n+k] * matrix[j*n+k];
			}
			spd_matrix[i*n+j] = sum;
		}
	}
	
	// check if all element in eigen vector are positives
	free(matrix);

    return spd_matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", name);

    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

int check_result(double *bref, double *b, double epsilon, int size) {
    int i;
    for(i=0;i<size*size;i++) {
        if ( abs(bref[i] - b[i]) > epsilon) 
			return 0;
    }
    return 1;
}


void cholesky (double *a, int size) {
    int i, j, k;
    double x, y;

	//#pragma omp parallel for
    for (j = 0; j < size; ++j) {
        x = 0.0;
        for (i = 0; i < j; ++i) {
            a[i+j*size] = 0.0;
            x += a[j+i*size] * a[j+i*size];
        }
        x = sqrt(a[j+j*size]-x);
        a[j+j*size] = x;
        for (i = j+1; i < size; ++i) {
            y = 0.0;
            for (k = 0; k < j; ++k)
                y += a[i+k*size] * a[j+k*size];
           a[i+j*size] = (a[i+j*size] - y) / x;
        }
    }
}

void solve (const double *a, double *b, int size) {
    int i, j, k;

    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            b[i+j*size] /= a[i+i*size];
            for (k = i+1; k < size; ++k)
                b[k+j*size] -= b[i+j*size] * a[k+i*size];
         }
    }
    for (j = 0; j < size; ++j) {
        for (i = size-1; i >= 0; --i) {
            for (k = i+1; k < size; ++k)
                b[i+j*size] -= a[k+i*size] * b[k+j*size];
            b[i+j*size] /= a[i+i*size];
        }
    }
}

int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
	cholesky(a, n);
	solve(a, b, n);
}

void my_dimatcopy(double *A, int n) {
	for (int i = 0; i < n; i++)
        for (int j = 0; j<i; j++) {
            double c = A[i*n + j];
            A[i*n + j] = A[j*n + i];
            A[j*n + i] = c;
        }
}

int main(int argc, char *argv[])
{
	int size = atoi(argv[1]);

	double *a, *aref;
	double *b, *bref;
	int info;
	
	a = generate_spd_matrix(size, &info);
	aref = generate_matrix(size);        
	b = generate_matrix(size);
	bref = generate_matrix(size);
	
	/*
	double a[25] = {3.14,   0.17,  -0.90,   1.65,  -0.72,
		 0.17,   0.79,   0.83,  -0.65,   0.28,
		 -0.90,   0.83,   4.53,  -3.70,   1.60,
		 1.65,  -0.65,  -3.70,   5.32,  -1.37,
		 -0.72,   0.28,   1.60,  -1.37,   1.98};
		 
	double aref[25];
	
	double b[25] = {
		 -7.29,   6.11,   0.59, 0, 0,
          9.25,   2.90,   8.88, 0, 0,
          5.99,  -5.05,   7.57, 0, 0,
          -1.94, -3.80,   5.57, 0, 0,
         -8.30,   9.66,  -1.67, 0, 0
	};
	
	double bref[25];  = {
           -7.29f,  9.25f,  5.99f, -1.94f, -8.30f,
            6.11f,  2.90f, -5.05f, -3.80f,  9.66f,
            0.59f,  8.88f,  7.57f,  5.57f, -1.67f,
			0,      0,     0,     0,   0,
			0,      0,     0,     0,   0
    };*/
	
	memcpy( aref, a, sizeof(double) * size * size);	
	memcpy( bref, b, sizeof(double) * size * size);	
	my_dimatcopy(bref, size);

	//print_matrix("A", a, size);
	//print_matrix("B", b, size);

	// Using MKL to solve the system
	MKL_INT n = size, nrhs = size, lda = size, ldb = size;
	MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

	clock_t tStart = clock();
	info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
	printf("Time taken by MKL: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	tStart = clock();    
	MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT)*size);        
	my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
	printf("Time taken by my implementation: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	
	// transpose result matrix from mkl function LAPACKE_dgesv
	my_dimatcopy(bref, size);
	
	double epsilon = 0.000001;
	if (check_result(bref, b, epsilon, size)==1)
		printf("Result is ok!\n");
	else    
		printf("Result is wrong!\n");
	
	//print_matrix("X", b, size);
	//print_matrix("Xref", bref, size);
	//print_matrix("aref", a, size);
	
	return 0;
}
