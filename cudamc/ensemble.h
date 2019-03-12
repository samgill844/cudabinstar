#include <curand.h>
#include <curand_kernel.h>

/*****************************************
*                GPU sampler             *
*****************************************/
__global__ void GPU_parallel_stretch_move_sampler(int nsteps, int ndim, int nwalkers, int blocks, int threads_per_block, double ** args, 
                                            double * loglikliehoods, double * positions,
                                            float a, curandState * devState);