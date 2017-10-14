#include "genresult.cuh"
#include <sys/time.h>


__global__ void Spmv(/*Arguments*/){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */
}
/*
__global__ void
spmv_csr_vector_kernel ( const int num_rows ,
                         const int * ptr ,
                         const int * indices ,
                         const float * data ,
                         const float * x ,
                         float * y)
{
    __shared__ float vals [];

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
    int warp_id = thread_id / 32; // global warp index
    int lane = thread_id & (32 - 1); // thread index within the warp

    // one warp per row
    int row = warp_id ;

    if ( row < num_rows ){
        int row_start = ptr [ row ];
        int row_end = ptr [ row +1];
        
        // compute running sum per thread
        vals [ threadIdx.x ] = 0;
        for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
            vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];

        // parallel reduction in shared memory
        if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
        if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
        if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
        if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
        if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];

        // segmented reduction in shared memory
        if( lane >= 1 && rows [ threadIdx.x ] == rows [ threadIdx.x - 1] )
            vals [ threadIdx.x ] += vals [ threadIdx.x - 1];
        if( lane >= 2 && rows [ threadIdx.x ] == rows [ threadIdx.x - 2] )
            vals [ threadIdx.x ] += vals [ threadIdx.x - 2];
        if( lane >= 4 && rows [ threadIdx.x ] == rows [ threadIdx.x - 4] )
            vals [ threadIdx.x ] += vals [ threadIdx.x - 4];
        if( lane >= 8 && rows [ threadIdx.x ] == rows [ threadIdx.x - 8] )
            vals [ threadIdx.x ] += vals [ threadIdx.x - 8];
        if( lane >= 16 && rows [ threadIdx.x ] == rows [ threadIdx.x - 16] )
            vals [ threadIdx.x ] += vals [ threadIdx.x - 16];
        

        // first thread writes the result
        if ( lane == 0)
            y[ row ] += vals [ threadIdx.x ];
        }
    }
    */
void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate things...*/
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int M = mat->M;
    int N = mat->N;
    int K = mat->nz;
    int *rIndex = mat->rIndex;
    int *cIndex = mat->cIndex;
    float *val= mat->val;

    int *nrIndex = (int *) malloc(K * sizeof (int)); 
    int *ncIndex = (int *) malloc(K * sizeof (int)); 
    float *nval = (float *) malloc(K * sizeof (float)); 

    int *rcount = (int *) malloc((M + 1) * sizeof (int)); 
    for (int i = 0; i < K; i++){
        rcount[rIndex[i]+1] += 1;
    }
    for (int i = 1; i < M+1; i++){
        rcount[i] += rcount[i-1]; 
    }

    for (int i = 0; i < K; i++){
            int k = rcount[rIndex[i]];
            while(nval[k] != 0){
                k++;
            }
            nval[k] = val[i];
            nrIndex[k] = rIndex[i];
            ncIndex[k] = cIndex[i];
    }
    for (int i = 0; i < 10; i ++){
        printf("rIndex[%d] = %d, cIndex[%d] = %d, val[%d] = %f \n", i, rIndex[i], i, cIndex[i], i, val[i]);
        printf("nrIndex[%d] = %d, ncIndex[%d] = %d, nval[%d] = %f \n", i, nrIndex[i], i, ncIndex[i], i, nval[i]);
    }
 
    //int W = blockSize * blockNum / 32;
    //int n_each_warp = (nz + W - 1) / W;
    /*
    int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/

    
    /*Invoke kernel(s)*/
    //Spmv<<<blockNum, blockSize>>>();

    //cudaDeviceSynchronize(); // this code has to be kept to ensure that all the kernels invoked finish their work
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

    /*Deallocate, please*/
}
