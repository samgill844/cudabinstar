#include <math.h>

__host__ __device__ double Fellipsoidal(double nu, double q, double radius_1, double incl, double u, double y)
{
    // SUrray of https://books.google.co.uk/books?id=ngtmDwAAQBAJ&pg=PA239&lpg=PA239&dq=ellipsoidal+variation+approximation+binary+star&source=bl&ots=swiO_JQdIR&sig=ACfU3U0HVtS8G37Z7EbdjDymUqICD36FgA&hl=en&sa=X&ved=2ahUKEwiO1tH9ud7hAhWDaFAKHRVoASIQ6AEwC3oECAkQAQ#v=onepage&q=ellipsoidal%20variation%20approximation%20binary%20star&f=false
    // q - mass ratio
    // radius_1 - R*/a 
    // incl - orbital inclination 
    // u - linear limb-darkening coefficient
    // y - gravitational darkening coefficient 
    // nu - true anomaly 
    //nu /= 2;
    // true anomaly goes from -pi to pi 
    // to convert to phase,we could ass pi and divide by two pi
    double alpha1 = ((y+2.)/(y+1.))*25.*u / (24.*(15. + u));
    double alpha2 = (y+1.)*(3.*(15.+u))/(20.*(3.-u));
    double Ae = alpha2*q*pow(radius_1,3)*pow(sin(incl),2); // Amplitude of ellipsoidal variation

    // Harmonic coeefs 
    double f1 = 3*alpha1*radius_1*(5*pow(sin(incl),2) - 4)/sin(incl);
    double f2 = 5*alpha1*radius_1*sin(incl);

    // Now return the variation 
    //return -Ae*( cos(2*M_PI*2*nu)    +    f1*cos(2*M_PI*nu)      +      f2*cos(2*M_PI*3*nu) );
    return -Ae*( cos(2*nu)    +    f1*cos(nu)      +      f2*cos(3*nu) );

}