#include <math.h> 


__host__ __device__ double Fdoppler(double alpha, double alpha_doppler, double K1 )
{
    double Ad = alpha_doppler*K1/299792.458;
    return Ad*sin(alpha-M_PI/2.);
}