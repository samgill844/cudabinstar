#include <math.h>
#include<stdlib.h>
#include<stdio.h>

#include "kepler.h"
#include "flux_drop.h"


extern "C" {__host__ __device__ void lc(const double * time, double * LC,
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

    for (i=0; i < N_LC; i++)
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
        LC[i] = l;
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

    double loglike=0.0, wt;

    for (i=0; i < N_LC; i++)
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

        // convert to mag
        l = zp - 2.5*log10(l);

        // Now do loglike
        wt = 1.0/(LC_ERR[i]*LC_ERR[i] + J*J);
        loglike -= 0.5*(pow(l - LC[i],2)*wt - log(wt));
    }
    return loglike;
}}