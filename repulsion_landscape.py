# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

@author: norepinefriend
"""

import numpy as np
import matplotlib.pyplot as plt

y0 = 1e-1
decay_const = 90
x = np.linspace(0,.15,200)
y = y0 * np.exp(-1 * decay_const * x)
plt.plot(x,y)
#plt.xlim(0,0.15)