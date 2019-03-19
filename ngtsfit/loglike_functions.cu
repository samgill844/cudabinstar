#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/lc.h"

/*************************/
/* Negative infinity for ln */
/*************************/
__device__ double neginfty(void)
{
	const long long ieee754inf =  0xfff0000000000000;
	return __longlong_as_double(ieee754inf);
}

__device__ __host__ double htoc1(double h1, double h2) {return 1 - h1 + h2;};

__device__ __host__ double htoc2(double c1, double h2){return log2(c1/h2);};

__device__ double lnlike( double * positions, int index3d, double ** args )
{
    /*             */
    // theta
    // -------
    // 0 : t_zero
    // 1 : period
    // 2 : radius_1
    // 3 : k
    // 4 : h1
    // 5 : h2
    // 6 : b
    // 7 : zp
    // 8 : J
    //
    // args
    // -------
    // 0 time
    // 1 LC
    // 2 LC_err
    // 3 N_LC
    // 4 t_zero_ref
    // 5 period_ref
    // h1_ref
    // h2_ref
    // 
    /*             */
    if ( positions[index3d+0] < args[3][1] -0.001*args[3][2]         ||  positions[index3d+0] > args[3][1]+0.001*args[3][2])          return neginfty();
    if ( positions[index3d+1] <  args[3][2]-0.001 ||  positions[index3d+1] > args[3][2]+0.001) return neginfty();
    if ( positions[index3d+2] < 0         ||  positions[index3d+2] > 0.8)         return neginfty();
    if ( positions[index3d+3] < 0         ||  positions[index3d+3] > 0.8)         return neginfty();
    if ( positions[index3d+4] < 0.4         ||  positions[index3d+4] > 0.9)         return neginfty();
    if ( positions[index3d+5] < 0.3         ||  positions[index3d+5] > 0.6)         return neginfty();
    if ( positions[index3d+6] < 0         ||  positions[index3d+6] > 1+positions[index3d+3])         return neginfty();
    if ( positions[index3d+8] < 0)                                                return neginfty();


    double incl = 180.*acos(positions[index3d+6]*positions[index3d+2])/M_PI;
    double ld1 = htoc1( positions[index3d+4], positions[index3d+5]);
    double ld2 = htoc2( ld1, positions[index3d+5]);

    double l =  lc_loglike(args[0], args[1], args[2], positions[index3d+7],positions[index3d+8],
        positions[index3d+0], positions[index3d+1],
        positions[index3d+2], positions[index3d+3] ,
        0., 0., 
        incl,
        0, ld1, ld2, 
        0., 0.,
        0, 0.001, 0, 0.001,
        (int) args[3][0] ) - 0.5*( pow(positions[index3d+4] - args[3][3], 2)/pow(0.003,2) + pow(positions[index3d+5] - args[3][4], 2)/pow(0.047,2));
    if (l==neginfty()) return neginfty();
    else return l;
}



double lnlikec( double * positions, int index3d, double ** args )
{
    /*             */
    // theta
    // -------
    // 0 : t_zero
    // 1 : period
    // 2 : radius_1
    // 3 : k
    // 4 : h1
    // 5 : h2
    // 6 : b
    // 7 : zp
    // 8 : J
    //
    // args
    // -------
    // 0 time
    // 1 LC
    // 2 LC_err
    // 3 N_LC
    // 4 t_zero_ref
    // 5 period_ref
    // h1_ref
    // h2_ref
    // 
    /*             */
    //printf("Positions[%d] = %f",index3d+0,positions[index3d+0]);
    
    if ( positions[index3d+0] < args[3][1] -0.001*args[3][2]         ||  positions[index3d+0] > args[3][1]+0.001*args[3][2])          return -INFINITY;
    if ( positions[index3d+1] <  args[3][2]-0.001 ||  positions[index3d+1] > args[3][2]+0.001) return -INFINITY;
    if ( positions[index3d+2] < 0         ||  positions[index3d+2] > 0.8)         return -INFINITY;
    if ( positions[index3d+3] < 0         ||  positions[index3d+3] > 0.8)         return -INFINITY;
    if ( positions[index3d+4] < 0.4         ||  positions[index3d+4] > 0.9)         return -INFINITY;
    if ( positions[index3d+5] < 0.3         ||  positions[index3d+5] > 0.6)         return -INFINITY;
    if ( positions[index3d+6] < 0         ||  positions[index3d+6] > 1+positions[index3d+3])         return -INFINITY;
    if ( positions[index3d+8] < 0)                                                return -INFINITY;


    double incl = 180.*acos(positions[index3d+6]*positions[index3d+2])/M_PI;

    
    double ld1 = htoc1( positions[index3d+4], positions[index3d+5]);
    double ld2 = htoc2( ld1, positions[index3d+5]);

    double l =  lc_loglike(args[0], args[1], args[2], positions[index3d+7],positions[index3d+8],
        positions[index3d+0], positions[index3d+1],
        positions[index3d+2], positions[index3d+3] ,
        0., 0., 
        incl,
        0, ld1, ld2, 
        0., 0.,
        0, 0.001, 0, 0.001,
        (int) args[3][0] ) - 0.5*( pow(positions[index3d+4] - args[3][3], 2)/pow(0.003,2) + pow(positions[index3d+5] - args[3][4], 2)/pow(0.047,2));
    if (l==-INFINITY) return -INFINITY;
    else return l;
}