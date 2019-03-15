#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

/*****************************************
*         Array index funcs              *
*****************************************/
__device__ __host__ int get_2D_index(int i, int j, int dj );

__device__ __host__ int get_3D_index(int i, int j, int k, int dj, int dk );

__device__ double lnlike( double * positions, int index3d, double ** args );

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*****************************************
*                GPU sampler             *
*****************************************/
__global__ void GPU_parallel_stretch_move_sampler(int nsteps, int ndim, int nwalkers, int blocks, int threads_per_block, double ** args, 
                                            double * loglikliehoods, double * positions,
                                            float a, curandState * devState,
                                            int * block_progress)
{
    int i,j,k, index, index2d, index2d_previous, index3d, index3d_previous, index3d_previous_ensemble, lower, block;
    double Z;

    // Get the walker ID
    j = blockIdx.x*blockDim.x + threadIdx.x;

    // First we need to evaluate the starting positions
    index2d = get_2D_index(0, j, nwalkers );
    index3d = get_3D_index(0, j , 0, nwalkers, ndim);
    loglikliehoods[index2d] = lnlike( positions, index3d, args );

    lower = (int) floorf(j/threads_per_block)*threads_per_block;
    block = (int) lower / threads_per_block;

    //printf("\nHello from walker %d", j);
    // iterate over steps
    for (i=1; i < nsteps; i++)
    {
        // Get the index for the other Ensemble
        if (j - lower > threads_per_block/2)  index = (int) (curand_uniform(&devState[j])*threads_per_block);
        else                 index = lower + (int) (curand_uniform(&devState[j])*threads_per_block);
        //synchronize the local threads writing to the local memory cache
        __syncthreads();

        // Now update the traisl position
        Z = pow(((a - 1.) * curand_uniform(&devState[j]) + 1.), 2.) /a ;

        for (k=0; k < ndim; k++) 
        {

                index3d = get_3D_index(i, j , k, nwalkers, ndim);
                index3d_previous = get_3D_index(i-1, j , k, nwalkers, ndim);
                index3d_previous_ensemble = get_3D_index(i-1, index , k, nwalkers, ndim);
   
                positions[index3d] = positions[index3d_previous_ensemble]  - Z*(positions[index3d_previous_ensemble] - positions[index3d_previous]);   
        }

        // Now evalute the trial positions
        index2d = get_2D_index(i, j, nwalkers );
        index2d_previous = get_2D_index(i-1, j, nwalkers );
        index3d = get_3D_index(i, j , 0, nwalkers, ndim);
        loglikliehoods[index2d] = lnlike( positions, index3d, args );

        // Assess trail position
        if (loglikliehoods[index2d] < loglikliehoods[index2d_previous])
        {
                //if ( curand_uniform(&devState[j]) > exp(loglikliehoods[index2d] - loglikliehoods[index2d_previous]))
              if ( curand_uniform(&devState[j]) > pow(Z, ndim-1)*exp(loglikliehoods[index2d] - loglikliehoods[index2d_previous]) )
                {
                    // Here, we got unlucky so revert
                    loglikliehoods[index2d] = loglikliehoods[index2d_previous];
                    for (k=0; k < ndim; k++) 
                    {
                            index3d = get_3D_index(i, j , k, nwalkers, ndim);
                            index3d_previous = get_3D_index(i-1, j , k, nwalkers, ndim);
                            positions[index3d] = positions[index3d_previous];
                    }
                }
        }  
        //if (j==lower && i%10==1) printf("\rBlock %02d at %5.2f ", block+1, 100.*(float) (i-1.0) / (float) nsteps);
        block_progress[block] = (int) (100. * (float) i / (float) nsteps) + 1;
     }
}




/*

int main()
{
       int nDevices;

       cudaGetDeviceCount(&nDevices);
       printf("Number of devices : %d\n\n",nDevices);
       for (int i = 0; i < nDevices; i++) {
              cudaDeviceProp prop;
              cudaGetDeviceProperties(&prop, i);
              printf("Device Number: %d\n", i);
              printf("  Device name: %s\n", prop.name);
              printf("  Memory Clock Rate (KHz): %d\n",
                     prop.memoryClockRate);
              printf("  Memory Bus Width (bits): %d\n",
                     prop.memoryBusWidth);
              printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
       }
       printf("\n\n");



       srand((unsigned int)time(NULL));

       double theta[2] = {0.47,-0.52};
       printf("Initial loglike : %f", lnlike(theta, 0));

       int nsteps = 10000;
       int ndim = 2;
       int nwalkers = 10240;

       // Make Arrays
       double * loglikliehoods;
       loglikliehoods = (double*) malloc ( sizeof(double)*nsteps*nwalkers);

       double * positions;
       positions = (double*) malloc ( sizeof(double)*nsteps*nwalkers*ndim);

       // Initialse the starting paremeters
       int j,k, index3d;
       for (j=0; j < nwalkers; j++)
       {
              for (k=0; k < ndim; k++)
              {
                     index3d = get_3D_index(0, j , k, nwalkers, ndim);
                     positions[index3d] = theta[k] + sampleNormal()*1e-6;
              }
       }

       // Run the sample
       //CPU_parallel_stretch_move_sampler(nsteps, ndim, nwalkers, loglikliehoods, positions, 2.0);


       free(loglikliehoods);
       free(positions);


    


       printf("\n\nNow on the GPU");
       printf("\nInitialising Curand states");
       fflush(stdout);

       // Initialise the curand states
       curandState *devState;
       gpuErrchk(cudaMalloc((void**)&devState, nwalkers*sizeof(curandState)));
       initCurand<<<ceil(nwalkers/256),256>>>(devState, 1);
       gpuErrchk(cudaGetLastError());


       gpuErrchk(cudaMalloc(&loglikliehoods, sizeof(double)*nsteps*nwalkers )); 
       gpuErrchk(cudaMalloc(&positions, sizeof(double)*nsteps*nwalkers*ndim )); 


       double * tmp;
       tmp = (double*) malloc ( sizeof(double)*nsteps*nwalkers*ndim);

       // Initialse the starting paremeters
       for (j=0; j < nwalkers; j++)
       {
              for (k=0; k < ndim; k++)
              {
                     index3d = get_3D_index(0, j , k, nwalkers, ndim);
                     tmp[index3d] = theta[k] + sampleNormal()*1e-3;
              }
       }
       printf("\nCopying starting positions over");fflush(stdout);
       gpuErrchk(cudaMemcpy(positions, tmp, nwalkers*ndim*sizeof(double), cudaMemcpyHostToDevice)); // dont need nsteps here


       int threads_per_block = 256;
       int nblocks = ceil(nwalkers / threads_per_block);
       printf("\nLaunching kenel with (%d, %d)... ", nblocks, threads_per_block);fflush(stdout);

       GPU_parallel_stretch_move_sampler<<<nblocks, threads_per_block>>>(nsteps, ndim, nwalkers, loglikliehoods, positions, 2.0, devState);
       gpuErrchk(cudaGetLastError());
       printf(" done!");fflush(stdout);


       printf("\nCopying back");fflush(stdout);
       gpuErrchk(cudaMemcpy(tmp, positions, nsteps*nwalkers*ndim*sizeof(double), cudaMemcpyDeviceToHost));


        // Then open output file
        FILE * fp;
        // open the file for writing
        fp = fopen ("output_gpu.dat","w");
        int i;
        printf("\nWriting out...");fflush(stdout);
        for (i=7000;i<nsteps;i++)
        {
            for (j=0; j<nwalkers ; j++)
            {
                index3d = get_3D_index(i, j , 0, nwalkers, ndim);
                fprintf(fp, "\n%d,%d,%.5f,%.5f", i, j, tmp[index3d], tmp[index3d+1] );
                //printf("\n%d,%d,%.5f,%.5f", i, j, tmp[index3d], tmp[index3d+1] );

            }
        }
        fclose(fp);
        printf(" Complete.\n\n");fflush(stdout);



       // Finally, free
       cudaFree(devState);
       cudaFree(loglikliehoods);
       cudaFree(positions);

}
*/