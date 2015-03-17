# -*- coding: utf-8 -*-
"""
Damped harmonic oscillator agent model

Motion for the damped oscillator is described as:

m*d^2x/dt^2 + 2*zeta*w0*(dx/dt) + w0^2 * x = 0

where x is the positon of the oscillator in one dimension, w0 i the frequency, 
zeta is the damping coefficient, and m is mass. 

w0 is the natural frequency of the harmonic oscillator, which is:
w0 = sqrt(k/m)

A second order ODE (odeint) is used to solve for the position of the mass, 
presently in 1D.

End goal: run in 2d, add driving force and bias.

Code modified from http://nbviewer.ipython.org/github/kialio/scientific-python-lectures/blob/master/Lecture-3-Scipy.ipynb
Variable names were changed to reflect terms in the original equations.

Created on Mon Mar 16 16:22:47 2015
@authors: Richard Decal, decal@uw.edu
        Sharri Zamore
        
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
from scipy.integrate import odeint, ode
from matplotlib import pyplot as plt
import numpy as np

#TODO: make user-input
m = 1   #mass
k = .02   #spring constant

#TODO: make Bias a class? Talk to Rich P about a smart way to make this object oriented.

#TODO: change output from y to another name, as we will need y for 2D solutions

def dy(y, t, zeta, w0):
    """
    The right-hand side of the damped oscillator ODE
    """
    x, dxdt = x0[0], x0[1]
    #y, dydt = y0[0], y0[1] #starting to think about 2D..
    
    #originally p = dx/dt, this was modified to include timesetp values
    #i feel like user-defined dt should be in the equations below...not sure
    
    dx = dxdt
    dxdot = (-2 * zeta * w0 * dxdt - w0**2 * x) / m

    return [dx, dxdot]
    
# initial state: 
x0 = [1.0, 0.0]
#y0 = [1.0 0.0]
#unclear of how this is determined.

# time coodinate to solve the ODE for
dt = 1 #timestep
runtime = 1000
num_dt = runtime/dt

t = np.linspace(0, num_dt, runtime)
w0 = np.sqrt(k/m)
zeta = 0;   #maybe rename this? I don't like using the same name for the ode input and the function 

#TODO: change the ode such that if zeta is an n x 1 array, it produces n solution arrays

# solve the ODE problem for three different values of the damping ratio
z1 = odeint(dy, x0, t, args=(zeta, w0)) # undamped

#let's leave the other variations out of this for now
#y2 = odeint(dy, y0, t, args=(0.2, w0)) # under damped
#y3 = odeint(dy, y0, t, args=(1.0, w0)) # critial damping
#y4 = odeint(dy, y0, t, args=(5.0, w0)) # over damped



fig, ax = plt.subplots()
ax.plot(t, z1[:,0], 'k', label="undamped", linewidth=0.25)
#plt.ylim(-2,2)

#ax.plot(t, y2[:,0], 'r', label="under damped")
#ax.plot(t, y3[:,0], 'b', label=r"critical damping")
#ax.plot(t, y4[:,0], 'g', label="over damped")
#ax.legend();

