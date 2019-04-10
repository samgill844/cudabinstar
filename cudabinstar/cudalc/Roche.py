import numpy as np 
import numba 
import matplotlib.pyplot as plt 

# Lets start with a spherical example 
theta = np.linspace(0,2*np.pi, 20)    # the angle around the star
dtheta = theta[1] - theta[0]

radii = np.linspace(0,1,100)                             # the radius of the star 
dr = radii[1] - radii[0] 

# limb darkening law 
alpha = 0.8
c = 0.8 

@numba.njit 
def ld_law(alpha, c, mu) : return 1 - c*(1 - mu**(alpha))

I_sum = 0


for i in range(theta.shape[0]):
    for j in range(radii.shape[0]):
        I_sum += dtheta*dr * ld_law(alpha, c, 1 - radii[j]**2)

