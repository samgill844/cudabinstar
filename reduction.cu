#include <math.h>
#include <time.h>
#include <stdio.h>
#include <cub/cub.cuh>

#define BLOCKSIZE  256


void swap(double* &a, double* &b){
    double *temp = a;
    a = b;
    b = temp;
  }

/**************************/
/* BLOCK REDUCTION KERNEL */
/**************************/
__global__ void sum(const double * __restrict__ indata, double * __restrict__ outdata, int N) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Specialize BlockReduce for type float. 
    typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduceT; 

    // --- Allocate temporary storage in shared memory 
    __shared__ typename BlockReduceT::TempStorage temp_storage; 

    double result;
    if(tid < N) result = BlockReduceT(temp_storage).Sum(indata[tid]);

    // --- Update block reduction value
    if(threadIdx.x == 0) outdata[blockIdx.x] = result;
}



int main()
{

    int N = 1048576;

    double *g_idata, *g_odata; 

    g_idata = (double *) malloc(N*sizeof(double));
    g_odata = (double *) malloc(N*sizeof(double));

    // now populate the in data 
    for (unsigned int i=1; i < N; i++) g_idata[i]=1.0*0.5;



    double sum_cpu=0., sum_gpu=0.;
    clock_t tic = clock();
    for(unsigned int i = 0; i < N; i++) sum_cpu += g_idata[i];
    clock_t toc = clock();

    printf("CPU time : %5.0f micro-seconds with sum = %f", 1E6*(double)(toc - tic) / CLOCKS_PER_SEC, sum_cpu);

    // Now define the GPU array 
    double * d_g_idata, * d_g_odata;
    cudaMalloc(&d_g_idata, N*sizeof(double));
    cudaMalloc(&d_g_odata, (N/ BLOCKSIZE)*sizeof(double));

    cudaMemcpy(d_g_idata, g_idata , N*sizeof(double), cudaMemcpyHostToDevice) ;

    int blocks = ceil((double) N/ (double) BLOCKSIZE);
    printf("\nblock1 = %d", blocks);
    tic = clock();

    do 
    {
        sum<<<blocks, BLOCKSIZE>>>(d_g_idata, d_g_odata, N);
        N = blocks; 
        blocks = ceil((double) N/ (double) BLOCKSIZE) ;
        printf("\n%d %d %d", N, blocks, N /BLOCKSIZE );
        swap(d_g_idata, d_g_odata);
    } while (N != 1 );
    cudaMemcpy(g_odata, d_g_idata, sizeof(double), cudaMemcpyDeviceToHost);
    toc = clock();




    printf("\nGPU time : %5.0f micro-seconds with sum = %f\n\n", 1E6*(double)(toc - tic) / CLOCKS_PER_SEC, g_odata[0]);


    //for(int i = 0; i < (N/BLOCKSIZE); i++)
    //{
    //    printf("\nblock %d sum = %f", i, g_odata[i]);
    //}

    free(g_idata);
    free(g_odata);

    cudaFree(d_g_idata);
    cudaFree(d_g_odata);

    return 0;

}