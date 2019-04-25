import numpy as np 
import numba 
import matplotlib.pyplot as plt 
from cudabinstar.cudalc import lc 


nu = np.linspace(0, 5*np.pi,1000)



a = 5.6 
e = 0.
w  = 3*np.pi/2
Rp =SBR= 0.2
r = a*(1-e**2) / (1 + e*np.cos(nu)) 
incl = np.pi/2 

alpha = np.arccos(np.sin(w + nu)*np.sin(incl)) 

g = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha))/np.pi 
Fr = (SBR)*(1)*g*Rp**2 / r**2 


Ft = lc(nu, SBR=SBR, period = 2*np.pi)

plt.plot(nu, Fr+Ft) 
plt.axhline(1., c='k', ls='--')
plt.show()
