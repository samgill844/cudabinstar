import numpy as np 
import numba 
import matplotlib.pyplot as plt 
from cudabinstar.cudalc import lc
# from https://books.google.co.uk/books?id=ngtmDwAAQBAJ&pg=PA239&lpg=PA239&dq=ellipsoidal+variation+approximation+binary+star&source=bl&ots=swiO_JQdIR&sig=ACfU3U0HVtS8G37Z7EbdjDymUqICD36FgA&hl=en&sa=X&ved=2ahUKEwiO1tH9ud7hAhWDaFAKHRVoASIQ6AEwC3oECAkQAQ#v=onepage&q=ellipsoidal%20variation%20approximation%20binary%20star&f=false

phase = np.linspace(-1,1,1000)
q = 0.1
radius_1 = 0.5 
incl = np.pi/2 

u = 0.5 #linear limb darkening 
y = 0.3 # gravity darkening



Ft = lc(phase, radius_1=0.2, k =0.3)

alpha1 = ((y+2)/(y+1))*25*u / (24*(15 + u))
alpha2 = (y+1)*(3*(15+u))/(20*(3-u))

Ae = alpha2*q*(radius_1**3)*np.sin(incl)**2
f1 = 3*alpha1*radius_1*(5*np.sin(incl)**2 - 4)/np.sin(incl) 
f2 = 5*alpha1*radius_1*np.sin(incl) 

Fe = -Ae*( np.cos(2*np.pi*2*phase)    +    f1*np.cos(2*np.pi*phase)      +      f2*np.cos(2*np.pi*3*phase) )
plt.plot(phase, -2.5*np.log10(Ft + Fe)) 
plt.gca().invert_yaxis()
plt.show()