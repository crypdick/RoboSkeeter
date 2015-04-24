# -*- coding: utf-8 -*-
"""
Plot Sharri's distriubtion data

Created on Fri Apr 24 12:21:55 2015

@author: richard
"""

import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns


script_dir = os.path.dirname(__file__)  #behavior/postion/
rel_data_path = "data/distributions/"
os.chdir(rel_data_path)

arrays = {}
for file in glob("*.csv"):
    abspath = os.path.abspath(file)
    filename, extension = os.path.splitext ( os.path.basename(abspath) )
    print abspath
    arrays["{0}".format(filename)] = np.loadtxt(open(file, "rb"), delimiter=",")
    
# Plot!
fig, axs = sns.plt.subplots(1, 1)#, tight_layout=True)


axs.plot(arrays["bins"], arrays["ctrl_full_hist"], label='full tunnel')#, color=sns.desaturate("blue", .4), lw=2, label='$\mathbf{\dot{x}}$')
axs.plot(arrays["bins"], arrays["ctrl_upwind_hist"], label='upwind half')
axs.plot(arrays["bins"], arrays["ctrl_downwind_hist"], label='downwind half')

fig.suptitle("Sharri's Position Distributions", fontsize=14)
axs.set_ylabel("Probabilities")
axs.set_xlabel("Croosswind Distributions")
axs.legend()

plt.savefig("Dicks distributions.png")