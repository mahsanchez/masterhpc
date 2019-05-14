/* compute pi using Monte Carlo method */
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "mkl.h"
#include <omp.h>

#define CHUNKSIZE      1000
/* message tags */
#define REQUEST  1
#define REPLY    2

int rands[CHUNKSIZE], ranks[1];

VSLStreamStatePtr _stream;

int main(int argc, char *argv[])
{
    int provided;
    int iter;
    int in, out, i, iters, max, ix, iy, done, temp;
    double x, y, Pi, error, epsilon;
    int numprocs, myrank, server, totalin, totalout, workerid;
    double time_st, time_end, elapsed;
    double maxtime,mintime,avgtime;

    double start, end;

    int request;
    MPI_Comm world, workers;
    MPI_Group world_group, worker_group;
    MPI_Status status;

    //MPI_Init(&argc, &argv);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    printf("Nivel proporcionado %d de %d, %d, %d, %d\n", provided, MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);

    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myrank);
    server = numprocs-1;        /* last proc is server */

    if (myrank == 0) {
        epsilon = 0.0001;
		auto seed = 1234;
		vslNewStream(&_stream, VSL_BRNG_MT19937, (int)seed); 
    }

    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Comm_group(world, &world_group);
    ranks[0] = server;
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
    MPI_Comm_create(world, worker_group, &workers);
    MPI_Group_free(&worker_group);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (myrank == server) {
        do {
            MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
            if (request) {
                for (i = 0; i < CHUNKSIZE; i++) {
                    rands[i] = rand();
                    //if (rands[i] <= INT_MAX) i++;
                }
                MPI_Send(rands, CHUNKSIZE, MPI_INT, status.MPI_SOURCE, REPLY, world);
            }
        }
        while(request > 0);
    }
    else {
        request = 1;
        done = in = out = 0;
        max  = INT_MAX;         /* max int, for normalization */
        MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
        MPI_Comm_rank(workers, &workerid);
        iter = 0;
        while (!done) {
            iter++;
            request = 1;
            MPI_Recv(rands, CHUNKSIZE, MPI_INT, server, REPLY, world, MPI_STATUS_IGNORE);

            #pragma omp parallel for default(none) private(i, x, y) shared(rands, max) reduction(+:in,out)
            for (i=0; i<CHUNKSIZE; i+=2) {
                x = (((double) rands[i])/max) * 2 - 1;
                y = (((double) rands[i+1])/max) * 2 - 1;
                if (x*x + y*y < 1.0)
                    in += 1;
                else
                    out += 1;
            }

            MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, workers);
            MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, workers);

            Pi = (4.0*totalin)/(totalin + totalout);
            error = fabs( Pi-3.141592653589793238462643);
            done = (error < epsilon || (totalin+totalout) > 100000000);
            request = (done) ? 0 : 1;
            if (myrank == 0) {
                printf( "\rpi = %23.20f", Pi );
                MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            }
            else {
                if (request)
                    MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            }
        }
        MPI_Comm_free(&workers);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (myrank == 0) {
        printf( "\npoints: %d\nin: %d, out: %d, <ret> to exit\n", totalin+totalout, totalin, totalout );
        printf("elapsed time %1f", end - start);
		vslDeleteStream(&_stream);
    }
    MPI_Finalize();
    return 0;
}
