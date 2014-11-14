# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:16:05 2014

@author: richard

sensor neuron that models sensor neurons from
Gingl et al 2004 doi:10.1152/jn.01152/jn.01164.2004

input:
sensory value/time, e.g.  CO2 intensity d/dt

output:
spike rate
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, seed
from matplotlib import cm

def alphameshex():
    fig = plt.figure()
    ax = fig.gca(projection='3d')               # to work in 3d
    plt.hold(True)
    
    ##make mesh transparent
    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas
    
    ##generate a scatterplot
    n = 100
    seed(0)                                     # seed let us to have a reproducible set of random numbers
    x=[random() for i in range(n)]              # generate n random points
    y=[random() for i in range(n)]
    z=[random() for i in range(n)]
    ax.scatter(x, y, z);                        # plot a 3d scatter plot   
    
    x_surf=np.arange(0, 1, 0.01)                # generate a mesh
    y_surf=np.arange(0, 1, 0.01)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = np.sqrt(x_surf+y_surf)             # ex. function, which depends on x and y
    
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label') 
    ax.plot_surface(x_surf, y_surf, z_surf, cmap=theCM);    # plot a 3d surface plot
    plt.show()


def exampleplane():
    point1  = np.array([0,0,80])
    normal1 = np.array([1,-2,1])
    
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d1 = -np.sum(point1*normal1)# dot product
    
    # create x,y
    xx, yy = np.meshgrid(range(30), range(30))
    
    # calculate corresponding z
    z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]
    
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx,yy,z1, color='cyan')
    plt.show()    

def Ginglplane():
    """F = -63.2415 + 3.7504 T + 326.8702 dT/dt 
    3.7504 T + 326.8702 dT/dt - F - 63.2415 = 0
    
    data is constrained: 
    T between 30-24
    dT/dt between +- 0.03
    F between 25- 55
    """
    point1  = np.array([29,0,50])
    normal1 = np.array([3.7504,326.8702, -1])
    
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
#    d1 = -np.sum(point1*normal1)# dot product 
    d1 = -63.2415
    
    # create x,y
    xx, yy = np.meshgrid(np.linspace(31,24.5,100), np.linspace(-0.02,0.02, 100)) #ys should be from -.03 to .03
    
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
    plt3d.plot_surface(xx,yy,z1, cmap=theCM)
    
#    #make a shdow of the plane on the wall
#    cset = plt3d.contour(xx,yy,z1, zdir='z',  cmap=theCM)
#    cset = plt3d.contour(xx,yy,z1, zdir='x',  cmap=theCM)
#    cset = plt3d.contour(xx,yy,z1, zdir='y',  cmap=theCM)
    
#    plt3d.set_xlabel('X')
#    plt3d.set_xlim(-24, 29)
#    plt3d.set_ylabel('Y')
#    plt3d.set_ylim(-0.3, 0.3)
#    plt3d.set_zlabel('Z')
#    plt3d.set_zlim(25, 50)
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
    alphameshex()
#    exampleplane()
    Ginglplane()
    gingl_neuron(intensity_ddt,temp)

if __name__ == "__main__":
    main()