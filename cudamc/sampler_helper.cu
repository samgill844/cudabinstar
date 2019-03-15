#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



/*************************/
/* Negative infinity for ln */
/*************************/
/*
__device__ double neginfty(void)
{

	const unsigned long long ieee754inf =  0xfff0000000000000;
	return __longlong_as_double(ieee754inf);
}
*/

/*****************************************
*         Array index funcs              *
*****************************************/
__device__ __host__ int get_2D_index(int i, int j, int dj )
{
       return i*dj + j;
}

__device__  __host__  int get_3D_index(int i, int j, int k, int dj, int dk )
{
       return i*dj*dk + j*dk + k;
}

double sampleNormal_d() {
       double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
       double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
       double r = u * u + v * v;
       if (r == 0 || r > 1) return sampleNormal_d();
       double c = sqrt(-2 * log(r) / r);
       return u * c;
   }
   

void create_starting_positions(const double * theta,
                             const int nsteps, const int ndim, const int nwalkers,
                             const int blocks, const int threads_per_block,
                             const double scatter, 
                             double * d_positions, double * d_loglikliehoods)
{

    // Then allocate a tempory host array to hold the starting
    // point of each block
    double * p0;
    p0 = (double *) malloc(threads_per_block*ndim*sizeof(double));

    int i,k;

    // Now create the starting position
    for (i=0; i < threads_per_block; i++)
    {
        for (k=0; k < ndim; k++)
            p0[i*ndim + k] = theta[k] + scatter*sampleNormal_d();
    }

    // Now copy the starting points to each block
    for (i=0; i < blocks; i++)
    {
        gpuErrchk(cudaMemcpy(&d_positions[i*threads_per_block*ndim], p0, threads_per_block*ndim*sizeof(double), cudaMemcpyHostToDevice));
    }
    // Free allocated array
    free(p0);
}


void write_out_results(const int burn_in, const int nsteps, const int ndim, const int nwalkers,
                    const int blocks, const int threads_per_block,
                    double * d_positions, double * d_loglikliehoods, const char * output_filename)
{
    

    // Then allocate a tempory host array to hold the starting
    // point of each block
    double * p0, *loglikliehoods;
    p0 = (double *) malloc(threads_per_block*ndim*sizeof(double));
    loglikliehoods = (double *) malloc(threads_per_block*sizeof(double));


    // Then open output file
    FILE * fp;
    fp = fopen (output_filename,"w");

    int i,j,k,l;
    printf("\n");
    printf("Progress : %d %d %.2f", 0, 0, 0.); fflush(stdout);
    // For each step, we are going to write out block-by-block
    for (i=burn_in; i < nsteps; i++)
    {
        // iterate over the steps
        for (j=0; j < blocks; j++)
        {
            // iterate over the blocks
            cudaMemcpy(p0, &d_positions[i*nwalkers*ndim + j*threads_per_block*ndim], threads_per_block*ndim*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(loglikliehoods, &d_loglikliehoods[i*nwalkers + j*threads_per_block], threads_per_block*sizeof(double), cudaMemcpyDeviceToHost);

            // Now iterate over each walker
            for (k=0;k<threads_per_block;k++)
            {
                fprintf(fp, "\n%d,%d,%d", i+1, j+1, j*threads_per_block + k +1);
                for (l=0;l<ndim;l++)
                    fprintf(fp, ",%f", p0[k*ndim + l]);
                fprintf(fp, ",%f", loglikliehoods[k]);

                printf("\rProgress : %2d %2d %2.2f", i+1, j+1, (float) k / (float) threads_per_block); fflush(stdout);
            }
        }
    }

    // The close the file
    fclose(fp);

    // Then free memory
    free(p0);
    free(loglikliehoods);
}


/*************************/
/* CURAND INITIALIZATION */
/*************************/
__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

void create_curand_states(curandState *devState, int nwalkers)
{
       // Initialise the curand states
       initCurand<<<ceil(nwalkers/256),256>>>(devState, 1);
       cudaGetLastError();
}


/*************************/
/* reset the sampler     */
/*************************/
__global__ void reset_sampler(double * d_positions, double * d_loglikliehoods,
                            int ndim, int nwalkers, int nsteps,
                            int blocks, int threads_per_block)
{
    // get the thread ID
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    // get th eindexes
    int in2_last = get_2D_index(nsteps-1, j, nwalkers );
    int in3_last = get_3D_index(nsteps-1, j , 0, nwalkers, ndim);
    int in2_first = get_2D_index(0, j, nwalkers );
    int in3_first = get_3D_index(0, j , 0, nwalkers, ndim);

    int i;
    for (i=0; i < ndim; i++)
        d_positions[in3_first] = d_positions[in3_last];
    d_loglikliehoods[in2_first] = d_loglikliehoods[in2_last];
}

/*************************/
/* Sampler progress     */
/*************************/
__global__ void sampler_progress(int blocks, int * block_progress)
{
    int sum =0,i;
    int cols = 8, count=0;
    int rows = (int) ceil( (float) blocks / (float) cols);
    // First create the space to print
    for (i = 0; i< rows; i++) 
        printf("\n");

    
    while (sum != 100*blocks)
    {
        // Busy the worker for 10000 cycles
        clock_t start = clock();
        clock_t now;
        for (;;) 
        {
            now = clock();
            clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
            if (cycles >= 1000000) 
            {
                break;
            }
        }

        ///printf("\rSum = %d out of %d", sum, 100*blocks);
        
        // First we have to go up the amount of rows
        for (i = 0; i< rows-1; i++) 
            printf("\033[1F");

        //for (i = 0; i< rows; i++) 
        //    printf("\n");
        
        
        // First sum up the blocks progress
        sum = 0;
        for (i=0; i < blocks; i++) sum += block_progress[i];

        count = 0;

        // Now print the progress of each block in rows of 8
        for (i=0; i < blocks; i++)
        {
            if (count==cols)
            {
                printf("\n"); // put us on the next line
                count = 0;
            }
            printf("[%3d] ",block_progress[i]);
            count ++;
        }
        
    }
    
}
