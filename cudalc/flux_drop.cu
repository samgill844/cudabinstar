#include <math.h>

extern "C" {__device__ __host__ double  clip(double a, double b, double c)
{
    if (a < b)   return b;
    else if (a > c)  return c;
    else          return a;
}}

extern "C" {__device__ __host__ double Flux_drop_analytical_uniform( double z, double k, double SBR, double f)
{
    if (z >= 1. + k)  return f;  	                    // no overlap
    if (z >= 1. &&  z <= k - 1.)  return 0.0;           // total eclipse of the star
    else if  (z <= 1. - k)  return f - SBR*k*k;         // planet is fully in transit		
    else  
    {   					                            // planet is crossing the limb
        double kap1 = acos(min((1. - k*k + z*z)/2./z, 1.));
        double kap0 = acos(min((k*k + z*z - 1.)/2./k/z, 1.));
        return f - SBR*  (k*k*kap0 + kap1 - 0.5*sqrt(max(4.*z*z - pow(1. + z*z - k*k, 2.), 0.)))/M_PI;
    }
}}


extern "C" {__device__ __host__ double q1(double z, double p, double c, double a, double g, double I_0)
{
	double zt = clip(abs(z), 0,1-p);
	double s = 1-zt*zt;
	double c0 = (1-c+c*pow(s,g));
	double c2 = 0.5*a*c*pow(s,(g-2))*((a-1)*zt*zt-1);
    return 1-I_0*M_PI*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*pow(s,(g-1)));
}}

extern "C" {__device__ __host__ double q2(double z, double p, double c, double a, double g, double I_0, double eps)
{
	double zt = clip(abs(z), 1-p,1+p);
	double d = clip((zt*zt - p*p + 1)/(2*zt),0,1);
	double ra = 0.5*(zt-p+d);
	double rb = 0.5*(1+d);
	double sa = clip(1-ra*ra,eps,1);
	double sb = clip(1-rb*rb,eps,1);
	double q = clip((zt-d)/p,-1,1);
	double w2 = p*p-(d-zt)*(d-zt);
	double w = sqrt(clip(w2,eps,1));
	double c0 = 1 - c + c*pow(sa,g);
	double c1 = -a*c*ra*pow(sa,(g-1));
	double c2 = 0.5*a*c*pow(sa,(g-2))*((a-1)*ra*ra-1);
	double a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra);
	double a1 = c1+2*c2*(zt-ra);
	double aq = acos(q);
	double J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*pow(p,4))*aq ;
	double J2 = a*c*pow(sa,(g-1))*pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*sqrt(clip(1-q*q,0.0,1.0)) );
	double d0 = 1 - c + c*pow(sb,g);
	double d1 = -a*c*rb*pow(sb,(g-1));
	double K1 = (d0-rb*d1)*acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*sqrt(clip(1-d*d,0.0,1.0));
	double K2 = (1/3)*c*a*pow(sb,(g+0.5))*(1-d);
	if (J1 > 1) J1 = 0;
    double FF =  1 - I_0*(J1 - J2 + K1 - K2);
    if (FF < 0.9) FF=1.0;
    return FF;
}}


extern "C" {__device__ __host__ double  Flux_drop_analytical_power_2(double z, double k, double c, double a, double f, double eps)
{
    /*
    '''
    Calculate the analytical flux drop por the power-2 law.

    Parameters
    z : double
        Projected seperation of centers in units of stellar radii.
    k : double
        Ratio of the radii.
    c : double
        The first power-2 coefficient.
    a : double
        The second power-2 coefficient.
    f : double
        The flux from which to drop light from.
    eps : double
        Factor (1e-9)
    '''*/
    double I_0 = (a+2)/(M_PI*(a-c*a+2));
    double g = 0.5*a;

    if (z < 1-k)           return q1(z, k, c, a, g, I_0);
    else if (abs(z-1) < k) return q2(z, k, c, a, g, I_0, eps);
    else                   return 1.0;
}}









extern "C" {__device__ __host__ double ellpic_bulirsch(double n, double k)
{
    double kc = sqrt(1.-k*k);
    double p = sqrt(n + 1.);
    double m0 = 1.;
    double c = 1.;
    double d = 1./p;
    double e = kc;
    double f, g;

    int nit = 0;

    while(nit < 10000)
    {
        f = c;
        c = d/p + c;
        g = e/p;
        d = 2.*(f*g + d);
        p = g + p;
        g = m0;
        m0 = kc + m0;
        if(fabs(1.-kc/g) > 1.0e-8)
        {
            kc = 2.*sqrt(e);
            e = kc*m0;
        }
        else
        {
            return 0.5*M_PI*(c*m0+d)/(m0*(m0+p));
        }
        nit++;
    }
    return 0;
}}

extern "C" {__device__ __host__ double  ellec(double k)
{
    double m1, a1, a2, a3, a4, b1, b2, b3, b4, ee1, ee2, ellec;
    // Computes polynomial approximation for the complete elliptic
    // integral of the first kind (Hasting's approximation):
    m1 = 1.0 - k*k;
    a1 = 0.44325141463;
    a2 = 0.06260601220;
    a3 = 0.04757383546;
    a4 = 0.01736506451;
    b1 = 0.24998368310;
    b2 = 0.09200180037;
    b3 = 0.04069697526;
    b4 = 0.00526449639;
    ee1 = 1.0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)));
    ee2 = m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))*log(1.0/m1);
    ellec = ee1 + ee2;
    return ellec;
}}

extern "C" {__device__ __host__ double  ellk(double k)
{
    double a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, ellk,  ek1, ek2, m1;
    // Computes polynomial approximation for the complete elliptic
    // integral of the second kind (Hasting's approximation):
    m1 = 1.0 - k*k;
    a0 = 1.38629436112;
    a1 = 0.09666344259;
    a2 = 0.03590092383;
    a3 = 0.03742563713;
    a4 = 0.01451196212;
    b0 = 0.5;
    b1 = 0.12498593597;
    b2 = 0.06880248576;
    b3 = 0.03328355346;
    b4 = 0.00441787012;
    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)));
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4))))*log(m1);
    ellk = ek1 - ek2;
    return ellk;
}}

extern "C" {__device__ __host__ double  Flux_drop_analytical_quadratic(double d, double p, double c1, double c2, double tol)
{
    /*'''
    Calculate the analytical flux drop from the quadratic limb-darkening law.

    Parameters
    d : double
        Projected seperation of centers in units of stellar radii.
    p : double
        Ratio of the radii.
    c : double
        The first power-2 coefficient.
    a : double
        The second power-2 coefficient.
    f : double
        The flux from which to drop light from.
    eps : double
        Factor (1e-9)
    '''*/

    double kap0 = 0.0, kap1 = 0.0;
    double lambdad, lambdae, etad;
    double omega = 1.0 - c1/3.0 - c2/6.0;

    // allow for negative impact parameters
    d = fabs(d);

    // check the corner cases
    if(fabs(p - d) < tol)
    {
        d = p;
    }
    if(fabs(p - 1.0 - d) < tol)
    {
        d = p - 1.0;
    }
    if(fabs(1.0 - p - d) < tol)
    {
        d = 1.0 - p;
    }
    if(d < tol)
    {
        d = 0.0;
    }

    double x1 = pow((p - d), 2.0);
    double x2 = pow((p + d), 2.0);
    double x3 = p*p - d*d;

    //source is unocculted:
    if(d >= 1.0 + p)
    {
        //printf("zone 1\n");
        return 1.0;
    }
    //source is completely occulted:
    if(p >= 1.0 && d <= p - 1.0)
    {
        //printf("zone 2\n");
        lambdad = 0.0;
        etad = 0.5;        //error in Fortran code corrected here, following Jason Eastman's python code
        lambdae = 1.0;
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*(lambdad + 2.0/3.0) + c2*etad)/omega;
    }
    //source is partly occulted and occulting object crosses the limb:
    if(d >= fabs(1.0 - p) && d <= 1.0 + p)
    {
        //printf("zone 3\n");
        kap1 = acos(min((1.0 - p*p + d*d)/2.0/d, 1.0));
        kap0 = acos(min((p*p + d*d - 1.0)/2.0/p/d, 1.0));
        lambdae = p*p*kap0 + kap1;
        lambdae = (lambdae - 0.50*sqrt(max(4.0*d*d - pow((1.0 + d*d - p*p), 2.0), 0.0)))/M_PI;
    }

    //edge of the occulting star lies at the origin
    if(d == p)
    {
        //printf("zone 5\n");
        if(d < 0.5)
        {
            //printf("zone 5.2\n");
            double q = 2.0*p;
            double Kk = ellk(q);
            double Ek = ellec(q);
            lambdad = 1.0/3.0 + 2.0/9.0/M_PI*(4.0*(2.0*p*p - 1.0)*Ek + (1.0 - 4.0*p*p)*Kk);
            etad = p*p/2.0*(p*p + 2.0*d*d);
            return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
        }
        else if(d > 0.5)
        {
            //printf("zone 5.1\n");
            double q = 0.5/p;
            double Kk = ellk(q);
            double Ek = ellec(q);
            lambdad = 1.0/3.0 + 16.0*p/9.0/M_PI*(2.0*p*p - 1.0)*Ek -  \
                    (32.0*pow(p, 4.0) - 20.0*p*p + 3.0)/9.0/M_PI/p*Kk;
            etad = 1.0/2.0/M_PI*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 -  \
                            (1.0 + 5.0*p*p + d*d)/4.0*sqrt((1.0 - x1)*(x2 - 1.0)));
        //    continue;
        }
        else
        {
            //printf("zone 6\n");
            lambdad = 1.0/3.0 - 4.0/M_PI/9.0;
            etad = 3.0/32.0;
            return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
        }

        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
    }
    //occulting star partly occults the source and crosses the limb:
    //if((d > 0.5 + fabs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > fabs(1.0 - p)*1.0001 \
    //&& d < p))  //the factor of 1.0001 is from the Mandel/Agol Fortran routine, but gave bad output for d near fabs(1-p)
    if((d > 0.5 + fabs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > fabs(1.0 - p) \
        && d < p))
    {
        //printf("zone 3.1\n");
        double q = sqrt((1.0 - x1)/4.0/d/p);
        double Kk = ellk(q);
        double Ek = ellec(q);
        double n = 1.0/x1 - 1.0;
        double Pk = ellpic_bulirsch(n, q);
        lambdad = 1.0/9.0/M_PI/sqrt(p*d)*(((1.0 - x2)*(2.0*x2 +  \
                x1 - 3.0) - 3.0*x3*(x2 - 2.0))*Kk + 4.0*p*d*(d*d +  \
                7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk);
        if(d < p) lambdad += 2.0/3.0;
        etad = 1.0/2.0/M_PI*(kap1 + p*p*(p*p + 2.0*d*d)*kap0 -  \
            (1.0 + 5.0*p*p + d*d)/4.0*sqrt((1.0 - x1)*(x2 - 1.0)));
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
    }
    //occulting star transits the source:
    if(p <= 1.0  && d <= (1.0 - p))
    {
        etad = p*p/2.0*(p*p + 2.0*d*d);
        lambdae = p*p;

        //printf("zone 4.1\n");
        double q = sqrt((x2 - x1)/(1.0 - x1));
        double Kk = ellk(q);
        double Ek = ellec(q);
        double n = x2/x1 - 1.0;
        double Pk = ellpic_bulirsch(n, q);

        lambdad = 2.0/9.0/M_PI/sqrt(1.0 - x1)*((1.0 - 5.0*d*d + p*p +  \
                x3*x3)*Kk + (1.0 - x1)*(d*d + 7.0*p*p - 4.0)*Ek - 3.0*x3/x1*Pk);

        // edge of planet hits edge of star
        if(fabs(p + d - 1.0) <= tol)
        {
            lambdad = 2.0/3.0/M_PI*acos(1.0 - 2.0*p) - 4.0/9.0/M_PI* \
                        sqrt(p*(1.0 - p))*(3.0 + 2.0*p - 8.0*p*p);
        }
        if(d < p) lambdad += 2.0/3.0;
    }
    return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega;
}}
