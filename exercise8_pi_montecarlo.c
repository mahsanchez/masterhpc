/* compute pi using Monte Carlo method */
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define CHUNKSIZE      1000
#define CHUNKSIZE2     2000
/* message tags */
#define REQUEST  1
#define REPLY    2

int main(int argc, char *argv[])
{
    int iter;
    int in, out, i, iters, max, ix, iy, ranks[1], done, temp;
    double x, y, Pi, error, epsilon;
    int numprocs, myid, server, totalin, totalout, workerid;
    int rands[CHUNKSIZE], request;
    MPI_Comm world, workers, workers1, workers2;
    MPI_Group world_group, worker_group, worker_group1, worker_group2;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
	
    server = numprocs-1;	/* last proc is server */
    if (myid == 0) {
		if (argc < 2) {
			fprintf(stderr, "Usage: %s epsilon\n", argv[0] );
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
        sscanf( argv[1], "%lf", &epsilon );
    }
	
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Comm_group(world, &world_group);
    ranks[0] = server;
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
	MPI_Comm_create(world, worker_group, &workers);
	
	int color = myid % 2;
	int recvcount = numprocs - 1;
	int *recvbuf = (int *) malloc(recvcount * sizeof(int));
	MPI_Allgather(&color, 1, MPI_INT, recvbuf, recvcount, MPI_INT, workers);
	
	int numodds = 0;
	int numevens = 0;
	
	for (int i = 0; i < recvcount; i++) {
		if (recvbuf[i] == 0) {
		    numevens++;
		}
		else {
			numodds++;
		}
	}
	
	int* odd_ranks = (int *) malloc( numodds * sizeof(int) );
	for (int i = 0; i < numodds; i++) {
		odd_ranks[i] = i*2 + 1;
	}
	
	int* even_ranks = (int *) malloc( numevens * sizeof(int) );
	for (int i = 0; i < numevens; i++) {
		even_ranks[i] = i*2;
	}
	
	MPI_Group_incl(worker_group, numodds, odd_ranks, &worker_group1);
	MPI_Group_incl(worker_group, numevens, even_ranks, &worker_group2);
	
    MPI_Comm_create(workers, worker_group1, &workers1);
	MPI_Comm_create(workers, worker_group2, &workers2);
	
    MPI_Group_free(&worker_group);
	MPI_Group_free(&worker_group1);
	MPI_Group_free(&worker_group2);
	
    if (myid == server) {	/* I am the rand server */
		do {
			MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
			if (request) {
				int chunksize = (status.MPI_SOURCE % 2 == 0) ? CHUNKSIZE : CHUNKSIZE2;
				for (i = 0; i < chunksize; ) {
					rands[i] = random();
					if (rands[i] <= INT_MAX) i++;
				}
				MPI_Send(rands, chunksize, MPI_INT, status.MPI_SOURCE, REPLY, world);
			}
		}
		while(request > 0);
	}
    else {			/* I am a worker process */
		request = 1;
		done = in = out = 0;
		max  = INT_MAX;         /* max int, for normalization */
		MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
		//MPI_Comm_rank(workers, &workerid);
		
		iter = 0;
		while (!done) {
			iter++;
			request = 1;
			
			if (myid %2 == 0) {
				MPI_Recv(rands, CHUNKSIZE, MPI_INT, server, REPLY, world, MPI_STATUS_IGNORE);			
				for (i=0; i<CHUNKSIZE; ) {
					x = (((double) rands[i++])/max) * 2 - 1;
					y = (((double) rands[i++])/max) * 2 - 1;
					if (x*x + y*y < 1.0)
						in++;
					else
						out++;
				}
				MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, workers1);
				MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, workers1);
			}
			else {
				MPI_Recv(rands, CHUNKSIZE2, MPI_INT, server, REPLY, world, MPI_STATUS_IGNORE);			
				for (i=0; i<CHUNKSIZE2; ) {
					x = (((double) rands[i++])/max) * 2 - 1;
					y = (((double) rands[i++])/max) * 2 - 1;
					if (x*x + y*y < 1.0)
						in++;
					else
						out++;
				}
				MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, workers2);
				MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, workers2);
			}
			
			Pi = (4.0*totalin)/(totalin + totalout);
			error = fabs( Pi-3.141592653589793238462643);
			done = (error < epsilon || (totalin+totalout) > 100000000);
			request = (done) ? 0 : 1;
			if (myid == 0) {
			   printf( "\rpi = %23.20f", Pi );
			   MPI_Send(&request, 1, MPI_INT, server, REQUEST,world);
			}
			else {
			if (request)
				MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			}
		}
		MPI_Comm_free(&workers);
    }

    if (myid == 0) {
        printf( "\npoints: %d\nin: %d, out: %d, <ret> to exit\n",
		totalin+totalout, totalin, totalout );
	getchar();
    }
    MPI_Finalize();
    return 0;
}
