# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:18:13 2015

@author: richard


"""


import trajectory_data_io
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
import os
from math import floor as floor
import numpy as np
import seaborn as sns
import pandas as pd


plt.style.use('ggplot')

# make list of all csvs in dir
def make_csv_name_list():
    from glob import glob
    
    dyn_traj_reldir = "data/dynamical_trajectories/"
    os.chdir(dyn_traj_reldir)
    
    yield [os.path.splitext(file)[0] for file in glob("*.csv")]
    os.chdir(os.path.dirname(__file__))


csv_list = make_csv_name_list()
os.chdir(os.path.dirname(__file__))

### old code to solve modes
#csv_lens = []
#for csv_fname in csv_list:
#    print csv_fname
#    os.chdir(os.path.dirname(__file__))
#    df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
#
#    csv_lens.append(len(df.index))
#
#for csv_fname in csv_list:
#    print csv_fname
#    df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
#    csv_lens[csv_fname] =len(df.index)
#
#plt.plot(csv_lens)
#plt.ylim([0, 5000])

# for now, grab just one trajectory
a_trajectory = csv_list[12] # 5, 12


df = trajectory_data_io.load_trajectory_dynamics_csv(a_trajectory)
interested_vals = ['velo_x', 'velo_y', 'velo_z', 'curve']
df = df[interested_vals]

window_len = int(floor( len(df.index)/ 5 ))

lags = 20
num_segments = len(df.index) - window_len

segment_pd = []

# loop through segments
for segment_index in range(num_segments):
    segment = df[segment_index:segment_index+window_len]
    
    # for each segment, run PACF and ACF for each feature
    data_matrix = np.zeros((2*len(interested_vals), lags+1))
    for label, col in segment.iteritems():
        colindex = {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3}[label]
        
#        try:
        col_acf = acf(col, nlags=lags)#, alpha=.05,  qstat= True)
        
        # store data
        data_matrix[colindex] = col_acf
        ## , acf_confint, acf_qstats, acf_pvals
        col_pacf = pacf(col, nlags=lags)
        data_matrix[colindex+len(interested_vals)] = col_pacf
        
    segment_pd.append(pd.DataFrame(dict(condition=[["control", "left_heater", "right_heater"][0]] * (lags+1),
                                        subj=["seg%d" % segment_index] * (lags+1),
                                        lags=range(lags+1),
                                        acf_velox=data_matrix[0],
                                        acf_veloy=data_matrix[1],
                                        acf_veloz=data_matrix[2],
                                        acf_curve=data_matrix[3],
                                        pacf_velox=data_matrix[4],
                                        pacf_veloy=data_matrix[5],
                                        pacf_veloz=data_matrix[6],
                                        pacf_curve=data_matrix[7]
                                        )))
                                        
segment_pd = pd.concat(segment_pd)

analysis_types = {'acf_velox': "ACF Velocity x", 'acf_veloy': "ACF Velocity y",\
             'acf_veloz': "ACF Velocity Z", 'acf_curve': "ACF curvature", \
            'pacf_velox': "PACF Velocity x", 'pacf_veloy': "PACF Velocity y", \
            'pacf_veloz': "PACF Velocity z", 'pacf_curve': "PACF curvature"}
            
for analysis, title in analysis_types.iteritems():
    conf_ints = [95, 68]
    sns.tsplot(segment_pd, time="lags", unit="subj", condition="condition", value=analysis,\
#                                            err_style="unit_traces") # uncomment for unit traces
                                            err_style="ci_band", ci = conf_ints) # uncomment for CIs
    sns.plt.title(title)
    sns.plt.ylabel("Correlation")
    sns.plt.ylim[-1, 1]
    plt.savefig("./correlation_figs/{data_name}/{label}.svg".format(label=analysis, data_name = a_trajectory), format="svg")
    sns.plt.show()
