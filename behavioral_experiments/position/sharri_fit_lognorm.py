__author__ = 'richard'
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:07:31 2015

@author: richard
"""

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt


# load csv values
csv = np.genfromtxt('data/distributions/accelerationmag_raw.csv', delimiter=",")
csv = csv.T

bin_edges = csv[0]
probabilities = csv[4]

# draw a bunch of random samples from the probability distribution
rand_samples = np.random.choice(bin_edges, 1000, p=probabilities/probabilities.sum())

# fit to lognormal
shape, loc, scale = stats.lognorm.fit(rand_samples, floc=0)
mu = np.log(scale) # Mean of log(X)
sigma = shape # Standard deviation of log(X)
geom_mean = np.exp(mu) # Geometric mean == median
geom_stdev = np.exp(sigma) # Geometric standard deviation
scale = 1.

# multiply whole dist by scalar 0.1
#shape/sigma, standard dev of log(X) =  0.932352661694 * 0.5
#Geometric std dev =  2.54047904005
#loc =  0
#scale =  0.666555094117
#mu = log(scale) = Mean of log(X) =  -0.405632480939
#Geometric mean / median =  0.666555094117
SCALAR = 0.1

# plot together
plt.plot(bin_edges, probabilities, lw=2, label='sharri data')
plt.plot(bin_edges, stats.lognorm.pdf(bin_edges, shape*0.5, loc=loc, scale=scale), 'r', linewidth=3, label='fit')
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
print "mu = log(scale) = Mean of log(X) = ", mu
print "Geometric mean / median = ", geom_mean

# Sharri results
# multiply whole dist by scalar 0.1
#shape/sigma, standard dev of log(X) =  0.932352661694 * 0.5
#Geometric std dev =  2.54047904005
#loc =  0
#scale =  0.666555094117 * 0.5
#mu = log(scale) = Mean of log(X) =  -0.405632480939
#Geometric mean / median =  0.666555094117

#==============================================================================
# Dickinson results:
# shape = 0.719736466122
# loc = 0
# scale = 1.82216219069
# mu = 0.600023812816
# sigma = 0.719736466122
# geom_mean = 1.82216219069
# geom_stdev = 2.05389186923
#==============================================================================

sigma = 0.932352661694 * 0.5
scale = 1.6 *  0.666555094117 

# test
x = np.linspace(0,10,200)
pdf = SCALAR * scale * (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        / (x * sigma * np.sqrt(2 * np.pi)))
#y = np.random.lognormal(mean=mu, sigma=sigma, size=scale)
plt.plot(x, pdf)
#plt.plot(x, y)

#y = np.random.lognormal(mean=mu, sigma=sigma, size=scale)
