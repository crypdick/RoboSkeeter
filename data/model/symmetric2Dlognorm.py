# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

@author: norepinefriend
"""

import numpy as np
from numpy import linalg, newaxis, random
from matplotlib import pyplot as plt

def gen_rand_vecs(dims):
    vecs = random.normal(size=dims)
    mags = linalg.norm(vecs, axis=-1)

    ends = vecs / mags[..., newaxis]
    return ends
    
#    figure, axis = pyplot.subplots()
#    plt.scatter(ends[0], ends[1])
#
#    pyplot.show()

def main():
    spots = []
    rf= 0.5
    for i in range(50000):
        ends = gen_rand_vecs(2)
        mu = 0.600023812816
        sigma = 0.719736466122
        scale = 1.82216219069
        mag = np.random.lognormal(mean=mu, sigma=sigma, size=scale)
        spots.append(mag * ends)
    spots = np.asarray(spots)
    
#    # Estimate the 2D histogram
    nbins = 250
    magic = 5.
    H, xedges, yedges = np.histogram2d(spots[:,0],spots[:,1],bins=nbins)#, range=[[-magic, magic], [-magic, magic]])
#    
#    # H needs to be rotated and flipped
#    H = np.rot90(H)
#    H = np.flipud(H)
    H = H.T
#    
#    # Mask zeros
#    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
#    
#    # Plot 2D histogram using pcolor
    fig2 = plt.figure()
#    plt.hist2d(spots[:,0], spots[:,1], bins = 50)
    plt.pcolormesh(xedges,yedges,H, cmap=plt.cm.Oranges)
    plt.axis([-magic, magic, -magic, magic])
    plt.title("radially-symmetric Lognormal distribution")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    cbar = plt.colorbar()
#    cbar.ax.set_ylabel('Counts')
    
    plt.savefig("2D lognorm distribution.png")
    plt.show()
    return spots


    

spots = main()