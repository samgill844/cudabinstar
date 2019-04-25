#include<math.h>

#include "kepler.h"
#include "rv.h"

extern "C" {__host__ __device__ void rv (const double * time, 
        double t_zero, double period, 
        double K1, double fs, double fc, 
        double V0, double dV0, double VS, double VC, 
        double radius_1, double k, double incl, int ld_law_1, double ldc_1_1, double ldc_1_2,
        int Accurate_t_ecl, double t_ecl_tolerance,  int Accurate_Eccentric_Anomaly, double E_tol,
        double * RV_,  int N_start, int RV_N)
{
    // unpack arguments
    double w = atan2(fs, fc);
    double e = fs*fs + fc*fc;
    double nu;
    incl = M_PI*incl/180.;

    int i;
    for (i=0; i < RV_N; i++)
    {
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );
        RV_[N_start*RV_N + i] = K1*(e*cos(w) + cos(nu + w)) + V0 + dV0*(time[i] - t_zero)  ;
    }
}}


extern "C" {__host__ __device__  double rv_loglike (const double * time, double * RV, double *RV_ERR, double J,
    double t_zero, double period, 
    double K1, double fs, double fc, 
    double V0, double dV0, double VS, double VC, 
    double radius_1, double k, double incl, int ld_law_1, double ldc_1_1, double ldc_1_2,
    int Accurate_t_ecl, double t_ecl_tolerance,  int Accurate_Eccentric_Anomaly, double E_tol,
    int RV_N)
{
    // unpack arguments
    double w = atan2(fs, fc);
    double e = fs*fs + fc*fc;
    double nu, RV_model, wt, loglike=0.0;
    incl = M_PI*incl/180.;

    int i;
    for (i=0; i < RV_N; i++)
    {
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol );
        RV_model = K1*(e*cos(w) + cos(nu + w)) + V0 + dV0*(time[i] - t_zero)  ;
        wt = 1./(J*J + RV_ERR[i]*RV_ERR[i]);
        loglike -= 0.5*( pow(RV_model - RV[i],2)*wt - log10(wt) );
    }
    return loglike;
}}
