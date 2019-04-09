import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import math, numba, numba.cuda

__all__ = ['brent']

###################################################
# Fortran conversions
###################################################
@numba.njit
def sign(a,b) : 
    if b >= 0.0 : return abs(a)
    return -abs(a)




###################################################
# Brent minimisation
###################################################
@numba.njit
def brent(func,x1,x2, z0, tol):
    # pars
    #tol = 1e-5
    itmax = 100
    eps = 1e-5

    a = x1
    b = x2
    c = 0.
    d = 0.
    e = 0.
    fa = func(a,z0)
    fb = func(b,z0)

    fc = fb

    for iter in range(itmax):
        if (fb*fc > 0.0):
            c = a
            fc = fa
            d = b-a
            e=d   

        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0*eps*abs(b)+0.5*tol
        xm = (c-b)/2.0
        if (abs(xm) <  tol1 or fb == 0.0) : return b

        if (abs(e) > tol1 and abs(fa) >  abs(fb)):
            s = fb/fa
            if (a == c):
                p = 2.0*xm*s
                q = 1.0-s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0))
                q = (q-1.0)*(r-1.0)*(s-1.0)
            
            if (p > 0.0) : q = - q
            p = abs(p)
            if (2.0*p < min(3.0*xm*q-abs(tol1*q),abs(e*q))):
                e = d
                d = p/q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d   

        a = b
        fa = fb      
         
        if( abs(d) > tol1) : b = b + d
        else : b = b + sign(tol1, xm)

        fb = func(b,z0)
    return 1


#https://comp-astrophys-cosmol.springeropen.com/articles/10.1186/s40668-015-0008-8
@numba.jit
def A(p,e,v):
    return (p**2*(1 + e)**4) / ((1 + e*np.cos(v))**3)

@numba.jit
def Ohm(r, theta, phi, q, p, e, v):
    p1 = 1/r 
    p2 = q*(1/np.sqrt(1 - 2*r*np.sin(theta)*np.cos(phi) + r**2) - r*np.sin(theta)*np.cos(phi)) 
    p3 = (q + 1)*r**2*np.sin(theta)**2/2. 
    A1 = A(p,e,v) 
    return p1 + p2 + A1*p3 

@numba.jit
def dOhm(r, z) : return Ohm(r, z[0], z[1], z[2], z[3], z[4], z[5]) - Ohm(z[6]-r, z[0], z[1], z[2], z[3], z[4], z[5])

# r = radius
# theta = theta 
# phi = phi
#  ^ spherical coordinates
# q = M2/M1
# p = omega_star/omega_binary
# e = eccentricity 
# v = true anomaly  
r = np.linspace(0.01, 2.99,1000)
z = np.array([0.,0.,0.2,0.,0.,0., 3])

#First, let's work out L1
for i in r : plt.scatter(i, dOhm(i,z), c='k', s=10)
L1 = brent(dOhm,0.5,2.5, z, 1e-9)
plt.axvline(L1)
plt.show() 


z[6] = L1

@numba.jit
def dOhm_single(r, z) : return z[6] - Ohm(r, z[0], z[1], z[2], z[3], z[4], z[5])# - Ohm(z[6]-r, z[0], z[1], z[2], z[3], z[4], z[5])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta = np.linspace(0, 2*np.pi,20)
phi = np.linspace(0,np.pi,20)

def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

# Now work out 
for i in range(theta.shape[0]):
    for j in range(phi.shape[0]):
        z[0] = theta[i] 
        z[1] = phi[j] 
        radius = brent(dOhm_single,0.5,2.5, z, 1e-9)
        ax.scatter(*polar2cart(radius, theta[i], phi[j]), c='k')
plt.show()






