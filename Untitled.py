# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:07:31 2015

@author: richard
"""
#spots = []
#rf= 4e-06
#for i in range(10000):
#    theta = np.random.uniform(high=2*np.pi)
##    mag = np.random.exponential(rf)
#    mag = 1.
#    spots.append(mag * np.array([np.cos(theta), np.sin(theta)]))
#    
## Estimate the 2D histogram
#nbins = 200
#H, xedges, yedges = np.histogram2d(spots[0],spots[1],bins=nbins)
#
## H needs to be rotated and flipped
#H = np.rot90(H)
#H = np.flipud(H)
#
## Mask zeros
#Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
#
## Plot 2D histogram using pcolor
#fig2 = plt.figure()
#plt.pcolormesh(xedges,yedges,Hmasked)
#plt.xlabel('x')
#plt.ylabel('y')
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Counts')

import numpy
from numpy import linalg, newaxis, random
from matplotlib import collections, pyplot

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
    for i in range(10000):
        ends = gen_rand_vecs(2)
        mag = np.random.exponential(rf)
#        mag = 1.
        spots.append(mag * ends)
    spots = np.asarray(spots)
    
#    # Estimate the 2D histogram
    nbins = 100
    magic = 0.5
    H, xedges, yedges = np.histogram2d(spots[:,0],spots[:,1],bins=nbins), range=[[-magic, magic], [-magic, magic]])
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
    plt.pcolormesh(xedges,yedges,H)

#    plt.xlabel('x')
#    plt.ylabel('y')
#    cbar = plt.colorbar()
#    cbar.ax.set_ylabel('Counts')
    plt.show()


    

main()