#include <curand.h>
#include <curand_kernel.h>

double sampleNormal_d();

/*****************************************
*         Array index funcs              *
*****************************************/
__device__ __host__ int get_2D_index(int i, int j, int dj );

__device__ __host__ int get_3D_index(int i, int j, int k, int dj, int dk );



/*************************/
/* Negative infinity for ln */
/*************************/
//__device__ double neginfty(void);


void create_starting_positions(const double * theta,
                             const int nsteps, const int ndim, const int nwalkers,
                             const int blocks, const int threads_per_block,
                             const double scatter, 
                             double * d_positions,
                             int cpuORgpu);

void write_out_results(const int burn_in, const int nsteps, const int ndim, const int nwalkers,
                    const int blocks, const int threads_per_block,
                    double * d_positions, double * d_loglikliehoods, const char * output_filename,
                    int cpuORgpu, int verbose_flag);


__global__ void initCurand(curandState *state, unsigned long seed);

void create_curand_states(curandState *devState, int nwalkers);



__global__ void reset_sampler(double * d_positions, double * d_loglikliehoods,
                            int ndim, int nwalkers, int nsteps,
                            int blocks, int threads_per_block);


__global__ void sampler_progress(int blocks, int * block_progress);