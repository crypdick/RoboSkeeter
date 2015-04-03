# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:17:13 2015

@author: richard
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
from matplotlib.patches import Rectangle
import seaborn as sns

#sns.set_palette("muted", 8)

import pickle
fname = './mozzie_histograms.pickle'

with open(fname) as f:
    mozzie_hists = pickle.load(f)

odor_off = mozzie_hists['odor_off_hists']
odor_on = mozzie_hists['odor_on_hists']

## dict structure
# 'acceleration'
#    'y'
#    'x'
#    'z' 
#    'abs'
#'velocity'
#    'y'
#        'normed_cts', 'bin_centers'
#    'x'
#    'z'
#    'abs'
#'angular_velocity'
#    'y'
#    'x'
#    'z'
#    'abs'

#### Velocity
###v Odor off
fig, axs = sns.plt.subplots(2, 2, sharex=True, sharey=True)#, tight_layout=True)

#x
axs[0,0].plot(odor_off["velocity"]['x']['bin_centers'], odor_off["velocity"]['x']['normed_cts'], color=sns.desaturate("blue", .4), lw=3, label='$\mathbf{\dot{x}}$')
#y
axs[0,0].plot(odor_off["velocity"]['y']['bin_centers'], odor_off["velocity"]['y']['normed_cts'], color=sns.desaturate("green", .4), lw=3, label='$\mathbf{\dot{y}}$')
#z
axs[0,0].plot(odor_off["velocity"]['z']['bin_centers'], odor_off["velocity"]['z']['normed_cts'], color=sns.desaturate("yellow", .4), lw=3, label='$\mathbf{\dot{z}}$')
#abs
axs[0,0].plot(odor_off["velocity"]['abs']['bin_centers'], odor_off["velocity"]['abs']['normed_cts'], color=sns.desaturate("red", .4), lw=3, label='$\mathbf{\| a \|}$')

axs[0,0].set_title("Velocity distributions, Odor OFF")
axs[0,0].set_xlabel("Velocity")
axs[0,0].legend()  
#plt.savefig("./Agent Model/figs/DickinsonFigs/odorOFF_velo distributions.png")

###v Odor on
#x
axs[1,0].plot(odor_on["velocity"]['x']['bin_centers'], odor_on["velocity"]['x']['normed_cts'], color=sns.desaturate("blue", .4), lw=3, label='$\mathbf{\dot{x}}$')
#y
axs[1,0].plot(odor_on["velocity"]['y']['bin_centers'], odor_on["velocity"]['y']['normed_cts'], color=sns.desaturate("green", .4), lw=3, label='$\mathbf{\dot{y}}$')
#z
axs[1,0].plot(odor_on["velocity"]['z']['bin_centers'], odor_on["velocity"]['z']['normed_cts'], color=sns.desaturate("yellow", .4), lw=3, label='$\mathbf{\dot{z}}$')
#abs
axs[1,0].plot(odor_on["velocity"]['abs']['bin_centers'], odor_on["velocity"]['abs']['normed_cts'], color=sns.desaturate("red", .4), lw=3, label='$\| \mathbf{a} \|$')

axs[1,0].set_title("Velocity distributions, Odor ON")
axs[1,0].set_xlabel("Velocity")
axs[1,0].legend()
#plt.savefig("./Agent Model/figs/DickinsonFigs/odorON_velo distributions.png")


#### Acceleration
###a Odor off

#x
axs[0,1].plot(odor_off["acceleration"]['x']['bin_centers'], odor_off["acceleration"]['x']['normed_cts'], color=sns.desaturate("blue", .4), lw=3, label='$\mathbf{\ddot{x}}$')
#sns.barplot(odor_off["acceleration"]['x']['bin_centers'], odor_off["acceleration"]['x']['normed_cts'])
#y
axs[0,1].plot(odor_off["acceleration"]['y']['bin_centers'], odor_off["acceleration"]['y']['normed_cts'], color=sns.desaturate("green", .4), lw=3, label='$\mathbf{\ddot{y}}$')
#z
axs[0,1].plot(odor_off["acceleration"]['z']['bin_centers'], odor_off["acceleration"]['z']['normed_cts'], color=sns.desaturate("yellow", .4), lw=3, label='$\mathbf{\ddot{z}}$')
#abs
axs[0,1].plot(odor_off["acceleration"]['abs']['bin_centers'], odor_off["acceleration"]['abs']['normed_cts'], color=sns.desaturate("red", .4), lw=3, label='$\| \mathbf{a} \|$')
axs[0,1].set_title("Acceleration distributions, Odor OFF")
axs[0,1].set_xlabel("Acceleration")
plt.legend()  
#plt.savefig("./Agent Model/figs/DickinsonFigs/odorOFF_accel distributions.png")

###a Odor on
#x
axs[1,1].plot(odor_on["acceleration"]['x']['bin_centers'], odor_on["acceleration"]['x']['normed_cts'], color=sns.desaturate("blue", .4), lw=3, label='$\mathbf{\ddot{x}}$')
#sns.barplot(odor_off["acceleration"]['x']['bin_centers'], odor_off["acceleration"]['x']['normed_cts'])
#y
axs[1,1].plot(odor_on["acceleration"]['y']['bin_centers'], odor_on["acceleration"]['y']['normed_cts'], color=sns.desaturate("green", .4), lw=3, label='$\mathbf{\ddot{y}}$')
#z
axs[1,1].plot(odor_on["acceleration"]['z']['bin_centers'], odor_on["acceleration"]['z']['normed_cts'], color=sns.desaturate("yellow", .4), lw=3, label='$\mathbf{\ddot{z}}$')
#abs
axs[1,1].plot(odor_on["acceleration"]['abs']['bin_centers'], odor_on["acceleration"]['abs']['normed_cts'], color=sns.desaturate("red", .4), lw=3, label='$\| \mathbf{a} \|$')
axs[1,1].set_title("Acceleration distributions, Odor ON")
axs[1,1].set_xlabel("Acceleration")
axs[1,1].legend()
plt.savefig("./Agent Model/figs/DickinsonFigs/odorON_accel distributions.png")
