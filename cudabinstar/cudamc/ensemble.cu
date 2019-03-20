#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>
/*****************************************
*         Array index funcs              *
*****************************************/
__device__ __host__ int get_2D_index(int i, int j, int dj );

__device__ __host__ int get_3D_index(int i, int j, int k, int dj, int dk );

__device__ double lnlike( double * positions, int index3d, double ** args );
double lnlikec( double * positions, int index3d, double ** args );

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


/*****************************************
*                CPU sampler             *
*****************************************/
void CPU_parallel_stretch_move_sampler(int nsteps, int ndim, int nwalkers, double ** args, 
       double * loglikliehoods, double * positions,
       float a)
{
       int i,j,k, index, index2d, index2d_previous, index3d, index3d_previous, index3d_previous_ensemble;
       double Z;

       
       for (j=0; j<nwalkers ; j++)
       {
              // First we need to evaluate the starting positions
              index2d = get_2D_index(0, j, nwalkers );
              index3d = get_3D_index(0, j , 0, nwalkers, ndim);
              loglikliehoods[index2d] = lnlikec( positions, index3d, args ); 
              //if (j==0) printf("\nloglikliehoods[%d] = %f", index2d , loglikliehoods[index2d]) ;
       }
       // Then main loop
       
       
       for (i=1; i < nsteps; i++)
       {
              //#pragma omp parallel for shared(args, loglikliehoods, positions)
              for (j=0; j<nwalkers ; j++)
              {
                     // Get the index for the other Ensemble
                     if (j > nwalkers/2)    index = rand() % (nwalkers/2 - 0);
                     else                 index = nwalkers/2 +  rand() % (nwalkers - nwalkers/2);
                     Z = pow(((a - 1.) * (double) rand()/RAND_MAX + 1.), 2.) /a ;
                     //if (j==0) printf("\n%d %d %d ", i, index, nwalkers);
                     for (k=0; k < ndim; k++) 
                     {
                            index3d = get_3D_index(i, j , k, nwalkers, ndim);
                            index3d_previous = get_3D_index(i-1, j , k, nwalkers, ndim);
                            index3d_previous_ensemble = get_3D_index(i-1, index , k, nwalkers, ndim);
                            positions[index3d] = positions[index3d_previous_ensemble]  - Z*(positions[index3d_previous_ensemble] - positions[index3d_previous]);   
                            //if (j==0) printf("%f ", positions[index3d] );
                     }
                     //printf("\nloglikliehoods[%d] = %f", index2d , loglikliehoods[index2d]) ;

                     // Now evalute the trial positions
                     index2d = get_2D_index(i, j, nwalkers );
                     index2d_previous = get_2D_index(i-1, j, nwalkers );
                     index3d = get_3D_index(i, j , 0, nwalkers, ndim);
                     loglikliehoods[index2d] = lnlikec( positions, index3d, args ); 

                     //if (j==0) printf("\nloglikliehoods[index2d] , loglikliehoods[index2d_previous] = %f %f", loglikliehoods[index2d] , loglikliehoods[index2d_previous]) ;

                     // Assess trail position
                     if (loglikliehoods[index2d] < loglikliehoods[index2d_previous])
                     {
                            //if ( curand_uniform(&devState[j]) > exp(loglikliehoods[index2d] - loglikliehoods[index2d_previous]))
                            if ( (double) rand()/RAND_MAX > pow(Z, ndim-1)*exp(loglikliehoods[index2d] - loglikliehoods[index2d_previous]) )
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
              }
              printf("\rStep %4d out of %4d", i+1, nsteps);
       }
       
}