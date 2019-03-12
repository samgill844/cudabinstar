#include <math.h>
#define G 6.67408e-11


/************************************
*        Keplers equation           *
************************************/
__device__ __host__ double kepler (double M, double E, double e) { return M - E + e*sin(E);}
__device__ __host__ double dkepler (double E, double e) { return -1 + e*cos(E);}


/************************************
*        Eccentric Anomaly          *
************************************/
__device__ __host__ double getEccentricAnomaly (double M, double e, int Accurate_Eccentric_Anomaly, double tol)
{
    if (Accurate_Eccentric_Anomaly)
    {
        double m = fmod(M, 2*M_PI);
        int it;
        double f1 = fmod( m, (2*M_PI)) + e*sin(m) + e*e*sin(2.0*m)/2.0;
        double test=1.0;
        double e1,e0=1.0;
        while (test > tol)
        {
            it ++;
            e0=e1;
            e1 = e0 + (m-(e0 - e*sin(e0)))/(1.0 - e*cos(e0));
            test = abs(e1-e0);
        }

        if (e1 < 0) e1 = e1 + 2*M_PI;

        return e1;
    }
    else
    {
        if (e==0.) return M;

        int flip = 0;
        double m = fmod(M, (2*M_PI));
        if (m > M_PI)
        {
            m = 2*M_PI - m;
            flip = 1;
        }

        double alpha = (3*M_PI + 1.6*(M_PI-abs(m))/(1+e) )/(M_PI - 6/M_PI);
        double d = 3*(1 - e) + alpha*e;
        double r = 3*alpha*d * (d-1+e)*m + m*m*m;
        double q = 2*alpha*d*(1-e) - m*m;
        double w = pow((abs(r) + sqrt(q*q*q + r*r)),(2/3));
        double E = (2*r*w/(w*w + w*q + q*q) + m) / d;
        double f_0 = E - e*sin(E) - m;
        double f_1 = 1 - e*cos(E);
        double f_2 = e*sin(E);
        double f_3 = 1-f_1;
        double d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
        double d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3*d_3)*f_3/6);
        E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_4*d_4*d_4*f_2/24);

        if (flip==1) E = e*M_PI - E;
        return E;
    }
} 


/********************************************
*        Time of periastron passage         *
********************************************/
__device__ __host__ double t_ecl_to_peri(double t_ecl, double e, double w, double incl, double radius_1, double p_sid, double t_ecl_tolerance, int Accurate_t_ecl)
{
    // Define variables used
    //double efac  = 1.0 - e*2;
    double sin2i = pow(sin(incl),2);

    // Value of theta for i=90 degrees
    double ee, theta_0 = (M_PI/2) - w;             // True anomaly at superior conjunction

    //if (incl != math.pi/2.) and (Accurate_t_ecl == True) :  theta_0 =  brent(get_z_, theta_0-math.pi, theta_0 + math.pi,  (e, incl, w, radius_1), t_ecl_tolerance )
    if (theta_0 == M_PI)  ee = M_PI;
    else ee =  2.0 * atan(sqrt((1.-e)/(1.0+e)) * tan(theta_0/2.0));

    double eta = ee - e*sin(ee);
    double delta_t = eta*p_sid/(2*M_PI);
    return t_ecl  - delta_t;
}

/********************************************
*        Calculate the true anomaly         *
********************************************/
__host__ __device__ double getTrueAnomaly(double time, double  e, double w, double period, double t_zero, double incl, double radius_1, double t_ecl_tolerance, int Accurate_t_ecl,  int Accurate_Eccentric_Anomaly, double E_tol )
{
    // Sort inclination out
    incl = M_PI*incl / 180. ;

    // Calcualte the mean anomaly
    double M = 2*M_PI*fmod((time -  t_ecl_to_peri(t_zero, e, w, incl, radius_1, period, t_ecl_tolerance, Accurate_t_ecl)  )/period, 1.);

    // Calculate the eccentric anomaly
    double E = getEccentricAnomaly(M, e, Accurate_Eccentric_Anomaly, E_tol);

    // Now return the true anomaly
    return 2.*atan(sqrt((1.+e)/(1.-e))*tan(E/2.));
}

/***************************************************
*        Calculate the projected seperaton         *
***************************************************/
__device__ __host__ double get_z(double nu, double e, double incl, double w, double radius_1) {return (1-e*e) * sqrt( 1.0 - sin(incl)*sin(incl)  *  sin(nu + w)*sin(nu + w)) / (1 + e*sin(nu)) /radius_1;}
__device__ __host__ double get_z_(double nu, double * z){return get_z(nu, z[0], z[1], z[2], z[3]);}

/***************************************************
*        Calculate the projected position          *
***************************************************/
__device__ __host__ double getProjectedPosition(double nu, double w, double incl) {return sin(nu + w)*sin(incl);}

/*********************************************
*        Calculate the mass function         *
*********************************************/
__device__ __host__ double mass_function_1(double e, double P, double K1) {return pow(1-e*e,1.5)*P*86400.1* pow(K1*1000,3)/(2*M_PI*G*1.989e30);}
__device__ __host__ double mass_function_1_(double M2, double * z) {return (  pow(M2*sin(z[1]),3) / ( pow(z[0] + M2,2))) - mass_function_1(z[2], z[3], z[4]);}