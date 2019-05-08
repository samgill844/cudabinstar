#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cub/cub.cuh>

#include "kepler.h"
#include "flux_drop.h"
#include "ellipsoidal.h"
#include "reflected.h"
#include "doppler.h"
#include "spots.h"

#define BLOCKSIZE  256

extern "C" {__host__ __device__ double lc(const double * time, double * LC, double * LC_ERR, double J, double zp,
                        double t_zero, double period,
                        double radius_1, double k ,
                        double fs, double fc, 
                        double q, double albedo,
                        double alpha_doppler, double K1,
                        const double * spots, double omega_1, int nspots,
                        double incl,
                        int ld_law_1, double ldc_1_1, double ldc_1_2, double gdc_1,
                        double SBR, double light_3,
                        int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
                        int N_LC, int nthreads,
                        int logswitch )
{
    // Still do do 
    // Accoutn for difference in primary transit depth from SBR
    // Solve for linear limb-darkening coeff from limb-darkening law (maybe function pointers if supported?)
    int i,j;
    double nu, z, l, f;

    // Conversion
    double e = fs*fs + fc*fc;
    double w  = atan2(fs, fc);
    incl = M_PI*incl/180.;

    // Individual fluxes 
    double F_transit, F_doppler, F_ellipsoidal, F_reflected, F_spots, alpha;
    double u = 0.6;
    double loglike=0., wt;

#if !defined (__CUDACC__)
    omp_set_num_threads(nthreads);
    #pragma omp parallel for shared(LC, time, spots) private(nu, z, l, f, F_transit, F_doppler, F_ellipsoidal, F_reflected, F_spots, alpha, j) reduction(+:loglike)
#endif
    for (i=0; i < N_LC; i++)
    {
        // Get the true anomaly
        nu = getTrueAnomaly(time[i], e, w, period, t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );

        // Get the projected seperation
        z = get_z(nu, e, incl, w, radius_1);

        // Initialse the flux
        // The model we will use is:
        //   F_tot = F_ecl + F_d + F_ellipsoidal + F_spots + F_transit 
        //   F_ellipsoidal -> ellipsoidal effect contralled by the mass ratio, q and the gravity limb-darkening coefficient
        //   F_spots -> flux drop from spots controlled by the eker model
        //   F_transit -> flux from the desired transit model (using 1 as the continuum)
        l = 1.;
        F_transit=1;
        F_doppler = 0.;
        F_ellipsoidal=0;
        F_reflected=0;
        F_spots=0;

        // Check for stellar spots 
        if (nspots > 0)
        {
            double spot_phase = omega_1*2*M_PI*(time[i] - t_zero)/period;
            for (j=0; j < nspots; j++) F_spots += eker_spots(spots[j*4 + 0], spots[j*4 +1], incl, spots[j*4 +2], spots[j*4 +3], 0.5,0.3, spot_phase);
        }

        // Check for doppler beaming 
        if (alpha_doppler > 0 & K1 > 0)
        {
            //alpha = acos(sin(w + nu)*sin(incl));
            F_doppler = Fdoppler(nu, alpha_doppler, K1 );
        }

        // Check for eelipsoidal variation and apply it if needed
        if (q>0.)
        {
            alpha = acos(sin(w + nu)*sin(incl));
            F_ellipsoidal = Fellipsoidal(alpha, q, radius_1, incl, u, gdc_1);
        }; 

        // Check for reflected light from the secondary
        if (albedo > 0.)
        {
            alpha = acos(sin(w + nu)*sin(incl));
            F_reflected =  Freflected(alpha, nu, albedo, period, q, e, radius_1, k);
        }




        // Check distance between them to see if its transiting
        if (z < (1.0+ k))
        {
            // So it's eclipsing, lets find out if its a primary or secondary
            f = getProjectedPosition(nu, w, incl);

            if (f > 0)
            {
                if (ld_law_1==0) F_transit = Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, l, 1E-8);
                if (ld_law_1==1) F_transit = Flux_drop_analytical_quadratic(z, k, ldc_1_1, ldc_1_2, 1E-8);
            }
            else if (SBR>0.) F_transit =  Flux_drop_analytical_uniform(z, k, SBR, l); // Secondary eclipse
        }

        nu = F_transit + F_doppler+ F_spots + F_ellipsoidal + F_reflected; // we can re-cycle nu since we won't need it again in this loop.
        if (light_3 > 0.0)  nu = nu/(1. + light_3) + (1.-1.0/(1. + light_3)); // third light

        // Now do loglike if required
        if (logswitch)
        {
            nu = zp-2.5*log10(nu);
            wt = 1./(pow(LC_ERR[i], 2) + pow(J, 2));
            loglike += -0.5*( pow(LC[i] - nu, 2)*wt - log(wt)   );
        }
        else LC[i] = nu;
    }
    
    if (logswitch) return loglike; 
    else           return 0.;
}}






extern "C" {__global__ void d_lc(const double * time, double * LC, double * LC_ERR, double * loglike, double J, double zp,
                        double t_zero, double period,
                        double radius_1, double k ,
                        double fs, double fc, 
                        double q, double albedo,
                        double alpha_doppler, double K1,
                        const double * spots, double omega_1, int nspots,
                        double incl,
                        int ld_law_1, double ldc_1_1, double ldc_1_2, double gdc_1,
                        double SBR, double light_3,
                        int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
                        int N_LC )
{
    // Still do do 
    // Accoutn for difference in primary transit depth from SBR
    // Solve for linear limb-darkening coeff from limb-darkening law (maybe function pointers if supported?)
    int i = blockIdx.x*blockDim.x + threadIdx.x,j;
    double nu, z, l, f;

    // Conversion
    double e = fs*fs + fc*fc;
    double w  = atan2(fs, fc);
    incl = M_PI*incl/180.;

    // Individual fluxes 
    double F_transit, F_doppler, F_ellipsoidal, F_reflected, F_spots, alpha;
    double u = 0.6;
    double wt;


    // Get the true anomaly
    nu = getTrueAnomaly(time[i], e, w, period, t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );

    // Get the projected seperation
    z = get_z(nu, e, incl, w, radius_1);

    // Initialse the flux
    // The model we will use is:
    //   F_tot = F_ecl + F_d + F_ellipsoidal + F_spots + F_transit 
    //   F_ellipsoidal -> ellipsoidal effect contralled by the mass ratio, q and the gravity limb-darkening coefficient
    //   F_spots -> flux drop from spots controlled by the eker model
    //   F_transit -> flux from the desired transit model (using 1 as the continuum)
    l = 1.;
    F_transit=1;
    F_doppler = 0.;
    F_ellipsoidal=0;
    F_reflected=0;
    F_spots=0;

    // Check for stellar spots 
    if (nspots > 0)
    {
        double spot_phase = omega_1*2*M_PI*(time[i] - t_zero)/period;
        for (j=0; j < nspots; j++) F_spots += eker_spots(spots[j*4 + 0], spots[j*4 +1], incl, spots[j*4 +2], spots[j*4 +3], 0.5,0.3, spot_phase);
    }

    // Check for doppler beaming 
    if (alpha_doppler > 0 & K1 > 0)
    {
        //alpha = acos(sin(w + nu)*sin(incl));
        F_doppler = Fdoppler(nu, alpha_doppler, K1 );
    }

    // Check for eelipsoidal variation and apply it if needed
    if (q>0.)
    {
        alpha = acos(sin(w + nu)*sin(incl));
        F_ellipsoidal = Fellipsoidal(alpha, q, radius_1, incl, u, gdc_1);
    }; 

    // Check for reflected light from the secondary
    if (albedo > 0.)
    {
        alpha = acos(sin(w + nu)*sin(incl));
        F_reflected =  Freflected(alpha, nu, albedo, period, q, e, radius_1, k);
    }




    // Check distance between them to see if its transiting
    if (z < (1.0+ k))
    {
        // So it's eclipsing, lets find out if its a primary or secondary
        f = getProjectedPosition(nu, w, incl);

        if (f > 0)
        {
            if (ld_law_1==0) F_transit = Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, l, 1E-8);
            if (ld_law_1==1) F_transit = Flux_drop_analytical_quadratic(z, k, ldc_1_1, ldc_1_2, 1E-8);
        }
        else if (SBR>0.) F_transit =  Flux_drop_analytical_uniform(z, k, SBR, l); // Secondary eclipse
    }

    nu = F_transit + F_doppler+ F_spots + F_ellipsoidal + F_reflected; // we can re-cycle nu since we won't need it again in this loop.
    if (light_3 > 0.0)  nu = nu/(1. + light_3) + (1.-1.0/(1. + light_3)); // third light

    // Now do loglike if required
    nu = zp-2.5*log10(nu);
    wt = 1./(pow(LC_ERR[i], 2) + pow(J, 2));
    loglike[i] = -0.5*( pow(LC[i] - nu,2)*wt - log(wt)   );
    
}}

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

extern "C" {double lc_gpu(const double * time, double * LC, double * LC_ERR, double J, double zp,
    double t_zero, double period,
    double radius_1, double k ,
    double fs, double fc, 
    double q, double albedo,
    double alpha_doppler, double K1,
    const double * spots, double omega_1, int nspots,
    double incl,
    int ld_law_1, double ldc_1_1, double ldc_1_2, double gdc_1,
    double SBR, double light_3,
    int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
    int N_LC, int threads_per_block)
{

    // the first thing we will do is copy time, lightcurve and spot arrays to the GPU
    double * d_time, *d_LC, *d_LC_ERR, *d_spots, *d_loglike_in, *d_loglike_out; 

    int N_LC_LOGLIKE = pow(2,  ceil(log2(N_LC))  );
    //printf("\nN_LC = %d,    N_LC_LOGLIKE = %d", N_LC, N_LC_LOGLIKE);

    // now malloc
    cudaMalloc(&d_time, N_LC*sizeof(double));
    cudaMalloc(&d_LC, N_LC*sizeof(double));
    cudaMalloc(&d_LC_ERR, N_LC*sizeof(double));
    cudaMalloc(&d_loglike_in, N_LC_LOGLIKE*sizeof(double));  //
    cudaMalloc(&d_loglike_out, N_LC_LOGLIKE*sizeof(double)); // we need to allocate these two arrays the nearest power of 2 up for the reduction later
    cudaMalloc(&d_spots, 4*nspots*sizeof(double));

    // Now copy over the time, LC, LC_ERR and spot arrays 
    cudaMemcpy(d_time,   time, N_LC*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LC, LC, N_LC*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LC_ERR, LC_ERR, N_LC*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots, spots, 4*nspots*sizeof(double), cudaMemcpyHostToDevice);

    // The plan is:
    // 1. Launch the kernel which will populate d_loglike_in with the loglikes. 
    // 2. Recursively launch the reduction kernel to quickly sum it. This involves swapping device pointers. 
    // 3. Memcpy back to host and return the loglikliehood. 

    // memset the other loglike values to 0. 
    cudaMemset(&d_loglike_in[N_LC], 0.0, (N_LC_LOGLIKE - N_LC)*sizeof(double) );
    cudaMemset(&d_loglike_out[N_LC], 0.0, (N_LC_LOGLIKE - N_LC)*sizeof(double) );

    int blocks = ceil(N_LC/threads_per_block);

    // 1. Populate d_loglike_in with the loglikeliehood values at eachtime stamp
    d_lc<<<blocks, threads_per_block>>>(d_time, d_LC, d_LC_ERR, d_loglike_in, J, zp,
        t_zero, period,
        radius_1, k ,
        fs, fc, 
        q, albedo,
        alpha_doppler, K1,
        d_spots, omega_1, nspots,
        incl,
        ld_law_1, ldc_1_1, ldc_1_2, gdc_1,
        SBR, light_3,
        Accurate_t_ecl, t_ecl_tolerance, Accurate_Eccentric_Anomaly, E_tol,
        N_LC );



    // Wait for it to finish 
    cudaDeviceSynchronize();

    // 2. Recursively launch the reduction kernel until a lightucrve size of 1 is reched. 
    blocks = ceil(N_LC_LOGLIKE/BLOCKSIZE);
    //printf("\n%d %d %d", N_LC_LOGLIKE, blocks, N_LC_LOGLIKE /BLOCKSIZE );
    do 
    {
        sum<<<blocks, BLOCKSIZE>>>(d_loglike_in, d_loglike_out, N_LC);
        cudaDeviceSynchronize();

        N_LC_LOGLIKE = blocks; 
        blocks = ceil((double) N_LC_LOGLIKE/ (double) BLOCKSIZE) ;
        //printf("\n%d %d %d", N_LC_LOGLIKE, blocks, N_LC_LOGLIKE /BLOCKSIZE );
        swap(d_loglike_in, d_loglike_out);
    } while (N_LC_LOGLIKE != 1 );  

    //3. Now the first value of d_loglike_out should be the summed loglikliehood. 
    // Lets memcpy back 
    double loglike[1];
    cudaMemcpy(&loglike, d_loglike_in, sizeof(double), cudaMemcpyDeviceToHost);

    // finally free
    cudaFree(d_time);
    cudaFree(d_LC);
    cudaFree(d_LC_ERR);
    cudaFree(d_spots);
    cudaFree(d_loglike_in);
    cudaFree(d_loglike_out);

    return loglike[0];
}}