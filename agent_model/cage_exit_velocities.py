# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:46:48 2015

@author: richard

select box outside the cage
make histogram distributions
make polar plot of velocities


for each v,
    find magnitude, angle
    
the height should be sum of magnitudes / total magnitude

"""
import numpy as np
import matplotlib.pyplot as plt

# area bounds
# x:0.25 - 0.5
# y: all
door_region = ensemble.loc[((ensemble['position_x']>0.25) & (ensemble['position_x']<0.5)), ['velocity_x', 'velocity_y']]
door_region['magnitude'] = [np.linalg.norm(x) for x in door_region.values]
door_region['angle'] = np.tan(door_region['velocity_y']/door_region['velocity_x']) % (2*np.pi)
door_region['fraction'] = door_region['magnitude'] / door_region['magnitude'].sum()

N = 40  # number of bins


roundbins = np.linspace(0.0, 2 * np.pi, N)

width = (2*np.pi) / N

values, bin_edges = np.histogram(door_region['angle'], weights = door_region['fraction'], bins = roundbins)

ax = plt.subplot(111, polar=True)

ax.bar(bin_edges[:-1], values, width = width, linewidth = 0, alpha = 0.6)
plt.xlim(min(bin_edges), max(bin_edges))

# switch to radian labels
xT = plt.xticks()[0]
xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL, size = 20)
plt.title("Agent velocities 0.25 < x < 0.5, center repulsion on, n = 200", y=1.15)
plt.savefig("./figs/Velocity compass, center repulsion off, N200.svg", format='svg')


plt.show()