# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

@author: norepinefriend
"""

import numpy as np
import matplotlib.pyplot as plt

# exp params
y0 = 1e-1
decay_const = 90
# gaussian params
mu, stdev = 0, 0.5
x = np.linspace(-.15,0.15,200)
y = (y0 * np.exp(-1 * decay_const * (x+0.15))) + ( 1/(stdev * np.sqrt(2*np.pi)) * np.exp(-1*(x-mu)**2 / (2*stdev**2)) )  + (y0 * np.exp(1 * decay_const * (x-0.15)))
plt.plot(x,y)
#plt.xlim(0,0.15)


#y0 = 1
#decay_const = 90
#x = np.linspace(-.15,0.06,200)
##y = y0 * np.exp(-1 * decay_const * (x))
#plt.plot(x,y)
#y = y0 * np.exp(-1 * decay_const * (x+0.15))
#plt.plot(x,y, c='r')