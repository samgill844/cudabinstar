import numpy as np 
import numba 
import matplotlib.pyplot as plt 
from cudabinstar.cudalc import lc 


phase = np.linspace(-0.2,2.2,1000)
phase_ = phase*2*np.pi + np.pi

radius_1 = 0.2
k = 0.1
SBR=0.2

Ft = lc(phase, radius_1=radius_1, k = k, SBR=SBR) 

# from http://sf2a.eu/goutelas/2005/chap03-moutou.pdf
Ag = SBR # albedo 

Fr = 0.5*(1 + np.cos(phase_))*k**2*SBR

plt.plot(phase, Ft + Fr)
plt.show()