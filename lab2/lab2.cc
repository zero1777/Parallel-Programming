#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <mpi.h>

typedef unsigned long long ull;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	MPI_Init(&argc, &argv);
	int rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	ull result = 0;

	for (ull x = rank; x < r; x += world_size) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}
	pixels %= k;
	MPI_Reduce(&pixels, &result, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) printf("%llu\n", (4 * result) % k);

	MPI_Finalize();
}
