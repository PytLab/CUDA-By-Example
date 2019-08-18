#include "../common/book.h"

int main( void ) {
	cudaDeviceProp prop;

	int count;
	HANDLE_ERROR( cudaGetDeviceCount( &count ) );

	for( int i = 0; i < count; i++ ) {
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
		printf( "Name: %s\n", prop.name);
        printf( "Global Mem (Bytes): %lu\n", prop.totalGlobalMem );
        printf( "Shared Mem for each block (Bytes): %lu\n", prop.sharedMemPerBlock );
        printf( "32bit Register for each block: %d\n", prop.regsPerBlock );
        printf( "Warp size: %d\n", prop.warpSize );
        printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
        printf( "Max threads in every dim per block: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf( "Max blocks along every grid dim: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
        printf( "MultiProcess count:%d\n", prop.multiProcessorCount );
	}
}
