#include <math.h> 


__host__ __device__ void ld_quad_match(int ld_law, double * ldc, double * I_ret)
{
	// Set coefficients of a quadratic limb darkening law so that the intensity
	// profile matches at mu= 0, 0.5, 1.0
	// N.B.  these are the coefficients on the quadratic limb darkening law as used
	// in eker, i.e.,  I_0[1 - u_1.mu + u_2.mu^2], so u_2 is the negative of the
	// normal quadratic limb darkening coefficient.

	if (ld_law==-99)
	{
		I_ret[0] = 0.0;
		I_ret[1] = 0.0;
	}
	else if (ld_law==0)
	{
		I_ret[0] = ldc[0];
		I_ret[1] = 0.0;
	}
	else if (ld_law==1)
	{
		I_ret[0] = ldc[0];
		I_ret[1] = -ldc[1];		
	}
}




__host__ __device__ double eker_spots(double l, double b, double i, double r, double a, double u1, double u2, double phi)
{
    /*
    !
    ! Analytic light curve for a rotating star with a single circular spot 
    !
    ! Implementation of formulae from Eker, Z., 1994ApJ...420..373E with 
    ! correction for equation for K_3 from erratum 1994ApJ...430..438E.
    !
    ! Input values
    !  l        = longitude of spot centre (radians)
    !  b        = latitude of spot centre (radians)
    !  i        = inclination of rotation axis (radians)
    !  r        = angular radius of spot (radians)
    !  a        = Spot contrast ratio (a=Is/Ip).
    !  u1       = linear limb darkening coefficient.
    !  u2       = quadratic limb darkening coefficient.
    !  phi(nphi) = array of rotation phase values in radians, i.e. twopi*(t-t_0)/P
    !  nphi      = number of phase values
    !
    ! N.B. Eker uses a limb darkening law of the form I_0[1 - u_1.mu + u_2.mu^2], 
    ! i.e., u_2 is the negative of the normal quadratic limb darkening coefficient.
    !
    ! Output values
    !  df(nphi)  = light curve
    !  ii(nphi)  = spot position flag
    !  ifail    = status flag
    !
    ! Return values in the array df are (Fp + Fs)/F where
    !  Fp is the flux from the unspotted photosphere
    !  Fs is the flux from the spot
    !  F is the flux from the star without a spot.
    ! 
    ! Spot position flag ii() is as follows
    !   0 = spot not visible
    !   1 = Spot is on the limb and less than half the spot is visible
    !   2 = Spot is on the limb and more than half the spot is visible.
    !   3 = Spot is completely visible
    !
    ! Return value of ifail is the sum of the following flags.
    !   0 => all ok
    !   1 => r >= pi/2
    !   2 => a < 0.0
    !   4 => nphi < 1
    !
    */

    // Local variables 
    double cosisinb,sinicosb;
    double sinr2,sinr3,sinr4,sin2r;
    double cosr,cos2r,cosr3,cosr4;
    double qn,ql,qq;
    double th0,Cl,Cq,K1,K2,K3,K4;
    double t13b1,t13b2,t13c1,t13c2;
    double costh0,sinth0,sinth02,tanth0;
    double phi0,r0,sinphi0,cosphi0,sinr0,cosr0,sin2r0,cos2r0;
    double f0,fn,fl,fq;

    int ifail = 0;
    double pi = M_PI;
    double halfpi = M_PI/2.; 

    // First check
    if (r >= halfpi) ifail = ifail + 1;
    if (a < 0.0)  ifail = ifail + 2;
    if (ifail != 0) return 0. ;

    // constants 
    cosisinb = cos(i)*sin(b);
    sinicosb = sin(i)*cos(b);
    cosr = cos(r);
    cosr3 = pow(cosr,3);
    cosr4 = pow(cosr,4);
    sin2r = sin(2.0*r);
    cos2r = cos(2.0*r);
    sinr2 = 1.0 - pow(cosr,2);
    sinr3  = pow(sin(r),3);
    sinr4  = pow(sinr2,2);
    f0 = (a-1.0)/(1.0-u1/3.0+u2/6.0);
    fn = (1.0-u1+u2)*f0;
    fl = (u1-2.0*u2)*f0;
    fq = u2*f0;
    t13b1 = 2.0*(1.0-cosr3)/3.0;
    t13b2 = cosr*sinr2;
    t13c1 = 0.5*(1.0-cosr4);
    t13c2 = 0.75*sinr4;

    costh0 = cosisinb+sinicosb*cos(phi-l);
    th0 = acos(costh0);
    sinth02 = 1.0 - pow(costh0,2);
    sinth0 = sqrt(sinth02);
    tanth0 = sinth0/costh0;

    // First, check to make sure the spot isn't around the back 
    if (th0 >= (halfpi+r)) return 0.;
    else
    {
        // Here, the spot is partly or fully visible
        if (u1 == 0.0 & u2 == 0.0)
        {
            qn = sinr2*costh0;
            ql = 0.0;
            qq = 0.0;
        }
        else if (u2==0.)
        {
            qn = sinr2*costh0;
            ql = t13b1 - t13b2*sinth02;
            qq = 0.0;          
        }
        else
        {
            qn = sinr2*costh0;
            ql = t13b1 - t13b2*sinth02;
            qq = t13c1*pow(costh0,3) + t13c2*costh0*sinth02;
        }

        if (th0 > halfpi)
        {
            cosphi0 = -1.0/(tanth0*tan(r));
            phi0 = acos(cosphi0);
            sinphi0 = sin(phi0);
            qn = (phi0*qn - asin(cosr/sinth0) - 0.5*sinth0*sinphi0*sin2r)/pi + 0.5;
            
            if (u1 != 0.0 | u2 != 0.0)
            {
                r0 = abs(th0-halfpi);
                sinr0 = sin(r0);
                cosr0 = cos(r0);
                sin2r0 = sin(2.0*r0);
                cos2r0 = cos(2.0*r0);
                // (19a)
                ql = (phi0/3.0*(cosr3-pow(cosr0,3))* (1.0-3.0*pow(costh0,2)) - (phi0 + sinphi0*cosphi0)*(cosr - cosr0)*sinth02 + 4.0/3.0*sinphi0*(sinr3-pow(sinr0,3))*sinth0*costh0 + sinphi0*cosphi0/3.0*(cosr3 - pow(cosr0,3))*sinth02)/pi;
                if (u2 != 0.0) 
                {
                    K1 = 0.25*phi0*(pow(cosr0,4)-cosr4);
                    K2 = -0.125*sinphi0*(r0 - r + 0.5*(sin2r*cos2r-sin2r0*cos2r0));
                    K3 = 0.125*(phi0+sinphi0*cosphi0)*(sinr4-pow(sinr0,4));
                    K4 = (sinphi0-pow(sinphi0,3)/3.0)*(0.375*(r-r0) + 0.0625*(sin2r*(cos2r-4.0)-sin2r0*(cos2r0-4.0)));
                    qq = (2.0*pow(costh0,3)*K1 + 6.0*costh0*sinth0* (costh0*K2 + sinth0*K3) +  2.0*pow(sinth0,3)*K4 )/pi;     
                }  
            }
        } 
        else if (th0 > (halfpi-r)) // Spot is on the limb and more than half the spot is visible..
        {
            cosphi0 = -1.0/(tanth0*tan(r));
            phi0 = acos(cosphi0);
            sinphi0 = sin(phi0);
            qn = (phi0*qn - asin(cosr/sinth0) - 0.5*sinth0*sinphi0*sin2r)/pi + 0.5;

            if ((u1 != 0.0 | u2 != 0.0))
            {
                r0 = abs(th0-halfpi);
                sinr0 = sin(r0);
                cosr0 = cos(r0);
                sin2r0 = sin(2.0*r0);
                cos2r0 = cos(2.0*r0);
                // (18a)
                Cl = (pi-phi0)/3.0*(cosr3-pow(cosr0,3))*(1.0-3.0*pow(costh0,2)) - (pi-phi0-sinphi0*cosphi0)*(cosr-cosr0)*sinth02 - 4.0/3.0*sinphi0*(sinr3-pow(sinr0,3))*sinth0*costh0 - sinphi0*cosphi0*(cosr3-pow(cosr0,3))*sinth02/3.0;
                ql = ql - Cl/pi;
                if (u2 != 0.0)
                {
                    K1 = 0.25*(pi-phi0)*(pow(cosr0,4)-cosr4);
                    K2 = 0.125*sinphi0*(r0 - r + 0.5*(sin2r*cos2r-sin2r0*cos2r0));
                    K3 = 0.125*(pi-phi0+sinphi0*cosphi0)*(sinr4-pow(sinr0,4));
                    K4 = -(sinphi0-pow(sinphi0,3)/3.0)*(0.375*(r-r0) + 0.0625*(sin2r*(cos2r-4.0)-sin2r0*(cos2r0-4.0)));
                    // (18b)
                    Cq = 2.0*pow(costh0,3)*K1 + 6.0*costh0*sinth0*(costh0*K2 + sinth0*K3) + 2.0*pow(sinth0,3)*K4;
                    qq = qq - Cq/pi;
                }
            }
        }
        return fn*qn + fl*ql + fq*qq;
    }
}
