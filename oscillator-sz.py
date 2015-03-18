# -*- coding: utf-8 -*-
"""
Damped harmonic oscillator agent model

Motion for the damped oscillator is described as:

m*(d^2x/dt^2) + 2*zeta*w0*(dx/dt) + w0^2*x = 0

where x is the positon of the oscillator in one dimension, w0 i the frequency,
zeta is the damping coefficient, and m is mass.

w0 is the natural frequency of the harmonic oscillator, which is:
w0 = sqrt(k/m)

A second order ODE (odeint) is used to solve for the position of the mass,
presently in 1D.

End goal: run in 2d, add driving forces (random and temperature-stimulus bias).

Created on Mon Mar 16 16:22:47 2015
@authors: Richard Decal, decal@uw.edu
        Sharri Zamore

https://staff.washington.edu/decal/
https://github.com/isomerase/

Code forked from Lecture-3-Scipy.ipynb from the SciPy lectures:
https://github.com/jrjohansson/scientific-python-lectures
authored by J.R. Johansson (robert@riken.jp) http://dml.riken.jp/~rob/

Variable names have been changed to reflect terms in the original equations.
"""
from scipy.integrate import odeint, ode
from matplotlib import pyplot as plt
import numpy as np

# TODO: WTF: position stays constant, but acceleration changes? acceleration unbounded?

# CONSTANTS. TODO: make user-input. TODO: all caps for constant naming conventions
m = 1.0   # mass of agent
k = 0.02   # spring constant

# TODO: make Bias a class? Talk to Rich P about a smart way to make this object oriented.

# Initial state of the spring. 
x0 = [1.0, 0.0]  #position_0, velocity_0
#y0 = [1.0 0.0]  

# Time coodinates to solve the ODE for
dt = 1  # timebin width
runtime = 1000
num_dt = runtime/dt  # number of timebins
t = np.linspace(0, runtime, num_dt)


def dy(y, t, zeta, w0):
    """
    The right-hand side of the damped oscillator ODE
    (d^2x/dt^2) = ( 2*zeta*w0*(dx/dt) + w0^2*x ) / m
    """
    x, dxdt = x0[0], x0[1]
    #y, dydt = y0[0], y0[1] #starting to think about 2D..
    
    #originally p = dx/dt, this was modified to include timesetp values
    #i feel like user-defined dt should be in the equations below...not sure
    
    dx = dxdt #TODO wat?
    dxdot = (-2 * zeta * w0 * dxdt - w0**2 * x) / m  #dxdot -> x double dot?

    return [dx, dxdot] 

# TODO: change the ode such that if zeta is an n x 1 array, it produces n solution arrays

#ODE INPUTS
w0 = np.sqrt(k/m)
zeta_input = 0   # maybe rename? I don't like using the same name for the ode input and the function

# solve the ODE problem for three different values of the damping ratio
z1 = odeint(dy, x0, t, args=(zeta_input, w0))  # undamped

#let's leave the other variations out of this for now
#y2 = odeint(dy, y0, t, args=(0.2, w0))  # under damped
#y3 = odeint(dy, y0, t, args=(1.0, w0))  # critial damping
#y4 = odeint(dy, y0, t, args=(5.0, w0))  # over damped

fig, ax = plt.subplots()
ax.plot(t, z1[:, 0], 'k', label="undamped", linewidth=0.25)
#plt.ylim(-2,2)

#ax.plot(t, y2[:,0], 'r', label="under damped")
#ax.plot(t, y3[:,0], 'b', label=r"critical damping")
#ax.plot(t, y4[:,0], 'g', label="over damped")
#ax.legend();
