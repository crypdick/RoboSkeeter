# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:07:31 2015

@author: richard
"""

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

import Dickinson_experiments.dick_pickle


# grab odor off info
odor_off = Dickinson_experiments.dick_pickle.main(plotting=False)

binspots = odor_off["acceleration"]['abs']['bin_centers']
normed_counts = odor_off["acceleration"]['abs']['normed_cts']

# plot Dickinson acceleration magnitudes
#plt.plot(binspots, normed_counts, lw=2, label='Dickinson data')
#plt.legend()
#plt.show()

# normed_counts is set of probabilities, each one corresponding to a certain acceleration bin
# so, we must draw a bunch of random samples from the probability distribution
rand_samples = np.random.choice(binspots, 100000, p=normed_counts/normed_counts.sum())

# fit to lognormal
shape, loc, scale = stats.lognorm.fit(rand_samples, floc=0)
mu = np.log(scale) # Mean of log(X)
sigma = shape # Standard deviation of log(X)
geom_mean = np.exp(mu) # Geometric mean == median
geom_stdev = np.exp(sigma) # Geometric standard deviation

# Plot fit
#plt.plot(binspots, stats.lognorm.pdf(binspots, shape, loc=0, scale=scale), 'r', linewidth=3, label='fit')
#plt.legend()
#plt.show()

# plot together
#plt.plot(binspots, normed_counts, lw=2, label='Dickinson data')
plt.plot(binspots, stats.lognorm.pdf(binspots, shape, loc=0, scale=scale), 'r', linewidth=3, label='fit')
#plt.plot(binspots, stats.lognorm.pdf(binspots, shape, loc=0, scale=.1), 'g', linewidth=3, label='test')
plt.legend()
plt.show()

#plt.plot(binspots, stats.lognorm.pdf(binspots, shape, loc=0, scale=.1), 'g', linewidth=3, label='test')
#plt.xlim([0,1])
#plt.show()

# print results
print "shape/sigma, standard dev of log(X) = ", sigma
print "Geometric std dev = ", geom_stdev
print "loc = ", loc
print "scale = ", scale
print "mu / Mean of log(X) = ", mu
print "Geometric mean / median = ", geom_mean

#==============================================================================
# results:
# shape = 0.719736466122
# loc = 0
# scale = 1.82216219069
# mu = 0.600023812816
# sigma = 0.719736466122
# geom_mean = 1.82216219069
# geom_stdev = 2.05389186923
#==============================================================================

# test
x = np.linspace(0,10,200)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        / (x * sigma * np.sqrt(2 * np.pi)))
#y = np.random.lognormal(mean=mu, sigma=sigma, size=scale)
plt.plot(x, pdf)
#plt.plot(x, y)

#y = np.random.lognormal(mean=mu, sigma=sigma, size=scale)
