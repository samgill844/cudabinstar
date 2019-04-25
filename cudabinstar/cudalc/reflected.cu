#include <math.h> 

// From the exoplaent handbook 
// https://0-www-cambridge-org.pugwash.lib.warwick.ac.uk/core/services/aop-cambridge-core/content/view/643D8879D395AFA4B3F7F54A5E64BFEE/9781108304160c6_p153-328_CBO.pdf/transits.pdf 


__host__ __device__ double Freflected(double alpha, double nu, double albedo, double period, double q, double e, double radius_1, double k)
{
    // Calculate the orbital seperation, a, and the actual seperation, r
    double a = pow(  pow(period,2.)*8E-11*1.989E30*(1+q) / (4*pow(3.14159265359,2.) )      ,1./3.) / 695510E3; // in solar radii
    double r = a*(1-pow(e,2.)) / (1 + e*cos(nu));

    // Now calculate Rp/r = R*/r * Rp/R_*
    double r_star_over_r = radius_1*a/r; 
    double r_p_over_r = r_star_over_r*k; 

    // Now calculate the phase function describing the brightness of a reflecting 
    // object as a function of the phase angle
    alpha = M_PI - alpha;
    double g = (sin(alpha) + (M_PI - alpha)*cos(alpha))/M_PI; 

    // Now calculate the ratio of Rp^2/r^2 
    double ratio = pow(r_p_over_r, 2.);
    //printf("\n%f %f %f %f", a, r, g, ratio);

    return albedo*g*ratio;
}
