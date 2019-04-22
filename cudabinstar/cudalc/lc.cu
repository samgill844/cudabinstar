#include <math.h>
#include<stdlib.h>
#include<stdio.h>

#include "kepler.h"
#include "flux_drop.h"
#include "ellipsoidal.h"


extern "C" {__host__ __device__ void lc(const double * time, double * LC,
                        double t_zero, double period,
                        double radius_1, double k ,
                        double fs, double fc, double q,
                        double incl,
                        int ld_law_1, double ldc_1_1, double ldc_1_2, double gdc_1,
                        double SBR, double light_3,
                        int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
                        int N_LC )
{
    // Still do do 
    // Accoutn for difference in primary transit depth from SBR
    // Solve for linear limb-darkening coeff from limb-darkening law (maybe function pointers if supported?)
    int i;
    double nu, z, l, f;

    // Conversion
    double e = fs*fs + fc*fc;
    double w  = atan2(fs, fc);
    incl = M_PI*incl/180.;

    // Individual fluxes 
    double F_transit, F_ellipsoidal, F_spots;
    double u = 0.6;
    for (i=0; i < N_LC; i++)
    {
        // Get the true anomaly
        nu = getTrueAnomaly(time[i], e, w, period, t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );

        // Get the projected seperation
        z = get_z(nu, e, incl, w, radius_1);

        // Initialse the flux
        // The model we will use is:
        //   F_tot = F_ellipsoidal + F_spots + F_transit 
        //   F_ellipsoidal -> ellipsoidal effect contralled by the mass ratio, q and the gravity limb-darkening coefficient
        //   F_spots -> flux drop from spots controlled by the eker model
        //   F_transit -> flux from the desired transit model (using 1 as the continuum)
        l = 1.;
        F_transit=1;
        F_ellipsoidal=0;
        F_spots=0;

        // Check for eelipsoidal variation and apply it if needed
        if (q>0) F_ellipsoidal = Fellipsoidal(nu, q, radius_1, incl, u, gdc_1);



        

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

        // Create the total lightcurve 
        LC[i] = F_transit + F_spots + F_ellipsoidal;

        // Don't forget to account for third light
        if (light_3 > 0.0)  LC[i] = LC[i]/(1. + light_3) + (1.-1.0/(1. + light_3));
    }
}}


extern "C" {__host__ __device__ double lc_loglike(const double * time, const double * LC, const double * LC_ERR, double zp, double J,
    double t_zero, double period,
    double radius_1, double k ,
    double fs, double fc, 
    double incl,
    int ld_law_1, double ldc_1_1, double ldc_1_2, 
    double SBR, double light_3,
    int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
    int N_LC )
{
    int i;
    double nu, z, l, f;

    // Conversion
    double e = fs*fs + fc*fc;
    double w  = atan2(fs, fc);
    incl = M_PI*incl/180.;

    double loglike=0.0, wt, phase;
    double cut = radius_1*sqrt(pow(1 + k,2) + pow(cos(incl)/radius_1,2)) / M_PI;

    for (i=0; i < N_LC; i++)
    {
        phase =  (time[i] - t_zero) / period -  floor((time[i] - t_zero) /period);
        phase = phase > 0.5 ? -phase + 1. : phase;
        if (phase > cut & e==0 & SBR==0)
        {
            l = 1;
        }
        else
        {
            // Get the true anomaly
            nu = getTrueAnomaly(time[i], e, w, period, t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );

            // Get the projected seperation
            z = get_z(nu, e, incl, w, radius_1);

            // Initialse the flux
            l = 1.;

            // Check distance between them to see if its transiting
            if (z < (1.0+ k))
            {
                // So it's eclipsing, lets find out if its a primary or secondary
                f = getProjectedPosition(nu, w, incl);

                if (f > 0)
                {
                    if (ld_law_1==0) l = Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, l, 1E-8);
                    if (ld_law_1==1) l = Flux_drop_analytical_quadratic(z, k, ldc_1_1, ldc_1_2, 1E-8);
                }
                else if (SBR>0.) l =  Flux_drop_analytical_uniform(z, k, SBR, l); // Secondary eclipse

                // Don't forget to account for third light
                if (light_3 > 0.0)  l = l/(1. + light_3) + (1.-1.0/(1. + light_3));
            }
        }

        // convert to mag
        l = zp - 2.5*log10(l);

        // Now do loglike
        wt = 1.0/(LC_ERR[i]*LC_ERR[i] + J*J);
        loglike -= 0.5*(pow(l - LC[i],2)*wt - log(wt));
    }
    return loglike;
}}

/*
__global__ void  lc_kernel(const double * time, double * LC,
    double t_zero, double period,
    double radius_1, double k ,
    double fs, double fc, 
    double incl,
    int ld_law_1, double ldc_1_1, double ldc_1_2, 
    double SBR, double light_3,
    int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
    int N_LC )
{
    // Get the index
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    // Now laucn the kernel
    // We want 1 thread per LC
    void lc(time, &LC[j*N_LC],
        t_zero[j], period[j],
        radius_1[j], k[j] ,
        fs[j], fc[j], 
        incl[j],
        ld_law_1, ldc_1_1[j], ldc_1_2[j], 
        SBR[j], light_3[j],
        Accurate_t_ecl, t_ecl_tolerance,  Accurate_Eccentric_Anomaly, E_tol,
         N_LC )
}


extern "C" {__host__ __device__ void lc_batch(const double * time, double * LC,
    double * t_zero, double * period,
    double * radius_1, double * k ,
    double * fs, double * fc, 
    double * incl,
    int ld_law_1, double * ldc_1_1, double * ldc_1_2, 
    double * SBR, double * light_3,
    int Accurate_t_ecl, double t_ecl_tolerance, int Accurate_Eccentric_Anomaly, double E_tol,
    int N_LC, int nmodels, int threads_per_block )
{

    // Define pointers
    double *d_time;
    double *d_LC;
    double *d_t_zero;
    double *d_period;
    double *d_radius_1;
    double *d_k;
    double *d_k;
    double *d_k;
    double *d_k;
    double *d_k;
    double *d_k;
    double *d_k;
    double *d_k;


    // first malloc
    cudaMalloc(&d_time, N_LC*sizeof(double)); 
    cudaMalloc(&d_LC, N_LC*nmodels*sizeof(double)); 

    // then copy
    cudaMemcpy(d_time, time, N_LC*sizeof(double), cudaMemcpyHostToDevice);

    // Now execute
    lc_kernel<<<ceil(nmodels/threads_per_block), threads_per_block >>>(d_time, d_LC,
        t_zero, period,
        radius_1, k ,
        fs, fc, 
        incl,
        ld_law_1, ldc_1_1, ldc_1_2, 
        SBR, light_3,
        Accurate_t_ecl, t_ecl_tolerance, Accurate_Eccentric_Anomaly, E_tol,
        N_LC )

}}
*/