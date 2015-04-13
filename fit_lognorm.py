# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:07:31 2015

@author: richard
"""

from scipy import stats

plt.plot(odor_off["acceleration"]['abs']['bin_centers'], odor_off["acceleration"]['abs']['normed_cts'], color=sns.desaturate("black", .4), lw=2, label='$\| \mathbf{a} \|$')
plt.show()

#rvs = stats.lognorm.rvs(np.log(2), loc=0, scale=4, size=250) # Generate some random variates as data
#n, bins, patches = plt.hist(rvs, bins=25, normed=True)
#plt.show()

#odor_off["acceleration"]['abs']['bin_centers']
# odor_off["acceleration"]['abs']['normed_cts']
rvs = odor_off["acceleration"]['abs']['normed_cts']
shape, loc, scale = stats.lognorm.fit(rvs, floc=0)
mu = np.log(scale) # Mean of log(X)
sigma = shape # Standard deviation of log(X)
M = np.exp(mu) # Geometric mean == median
s = np.exp(sigma) # Geometric standard deviation

# Plot figure of results
x = np.linspace(rvs.min(), rvs.max(), num=400)
plt.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'r', linewidth=3)
#plt.plot()