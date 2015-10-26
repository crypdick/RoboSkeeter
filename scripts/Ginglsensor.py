# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:16:05 2014

@author: richard

sensor neuron that models sensor neurons from
Gingl et al 2004 doi:10.1152/jn.01152/jn.01164.2004

input:
sensory value/time, e.g.  temp intensity d/dt

output:
spike rate


TODO: need to verify that this is all reasonable

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def show_Gingl_plane():
    """F = -63.2415 + 3.7504 T + 326.8702 dT/dt 
    3.7504 T + 326.8702 dT/dt - F - 63.2415 = 0
    
    data is constrained: 
    T between 30-24
    dT/dt between +- 0.03
    F between 25- 55
    
    TODO: make sure that the surface code is properly implemented
    """
    point1  = np.array([29, 0, 50])
    normal1 = np.array([3.7504, 326.8702, -1])
    
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
#    d1 = -np.sum(point1*normal1)# dot product 
    d1 = -63.2415 # TODO: wtf?
    
    # create x,y
    xx, yy = np.meshgrid(np.linspace(31, 24.5,100), np.linspace(-0.02,0.02, 100)) #ys should be from -.03 to .03
    
    # calculate corresponding z
    z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]   
#    
#    z1 = normal1[0]*xx 
#    z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]
    
    ##make mesh transparent
    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas    
    
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z1, cmap=theCM)
    
    
    plt.show() 


def gingl_neuron(intensity_ddt,temp):
    """F = -63.2415 + 3.7504 T + 326.8702 dT/dt 
    3.7504 T + 326.8702 dT/dt - F - 63.2415 = 0
    """
    F = -63.2415 + 3.7504 * temp + 326.8702 * intensity_ddt
    print "fire rate at temp = " + str(temp) + " and d temp/dt = " + str(intensity_ddt) + " ==> " + str(F)


def main():  
    intensity_ddt = 0.0
    temp = 29.0
    show_Gingl_plane()
    gingl_neuron(intensity_ddt, temp)

if __name__ == "__main__":
    main()