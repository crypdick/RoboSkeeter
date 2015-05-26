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


INTERESTED_VALS = ['velo_x', 'velo_y', 'velo_z', 'curve']

#WINDOW_LEN = int(floor( len(df.index)/ 5 ))
WINDOW_LEN = 100
LAGS = 20
MIN_TRAJECTORY_LEN = 400


# make list of all csvs in dir
def make_csv_name_list():
    from glob import glob
    
    dyn_traj_reldir = "data/dynamical_trajectories/"
    os.chdir(dyn_traj_reldir)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__)) 
    return csv_list


### old code to solve modes
#csv_lens = []
#for csv_fname in csv_list:
#    print csv_fname
#    os.chdir(os.path.dirname(__file__))
#    df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
#
#    csv_lens.append(len(df.index))
#

#
#plt.plot(csv_lens)
#plt.ylim([0, 5000])

# for now, grab just one trajectory
#a_trajectory = csv_list[12] # 5, 12


#df = trajectory_data_io.load_trajectory_dynamics_csv(a_trajectory)

def loop_csvs(csv_list):
    segment_pd = []
    for csv_fname in csv_list:
        print csv_fname
        df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
        if len(df.index) < MIN_TRAJECTORY_LEN:
            pass
        else:
            df = df[INTERESTED_VALS] # slice only cols we want
            analysis_df = segment_analysis(df, csv_fname)
            segment_pd.append(analysis_df)
    
    return pd.concat(segment_pd)
            

def segment_analysis(trajectory_df, csv_fname):    
    num_segments = len(trajectory_df.index) - WINDOW_LEN
    
    # for each trajectory, loop through segments
    for segment_index in range(num_segments):
        segment = trajectory_df[segment_index:segment_index+WINDOW_LEN]
        
        # for each segment, run PACF and ACF for each feature
        data_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
        for label, col in segment.iteritems():
            colindex = {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3}[label]
            
    #        try:
            col_acf = acf(col, nlags=LAGS)#, alpha=.05,  qstat= True)
            
            # store data
            data_matrix[colindex] = col_acf
            ## , acf_confint, acf_qstats, acf_pvals
            col_pacf = pacf(col, nlags=LAGS)
            data_matrix[colindex+len(INTERESTED_VALS)] = col_pacf
            
        return pd.DataFrame(dict(#Condition=[["control", "left_heater", "right_heater"][0]] * (LAGS+1),
                                 Condition=[csv_fname] * (LAGS+1),
                                            Segment=["{0} seg{1:.2f}".format(csv_fname, segment_index)] * (LAGS+1),
                                            Lags=range(LAGS+1),
                                            acf_velox=data_matrix[0],
                                            acf_veloy=data_matrix[1],
                                            acf_veloz=data_matrix[2],
                                            acf_curve=data_matrix[3],
                                            pacf_velox=data_matrix[4],
                                            pacf_veloy=data_matrix[5],
                                            pacf_veloz=data_matrix[6],
                                            pacf_curve=data_matrix[7]
                                            ))
                                        
def plot_analysis(analysis_DF):
    analysis_types = {'acf_velox': "ACF Velocity x", 'acf_veloy': "ACF Velocity y",\
                 'acf_veloz': "ACF Velocity Z", 'acf_curve': "ACF curvature", \
                'pacf_velox': "PACF Velocity x", 'pacf_veloy': "PACF Velocity y", \
                'pacf_veloz': "PACF Velocity z", 'pacf_curve': "PACF curvature"}
                
    for analysis, title in analysis_types.iteritems():
        conf_ints = [95, 68]
        sns.tsplot(analysis_DF, time="Lags", unit="Segment", condition="Condition",\
                    color= sns.color_palette("Blues_r"), value=analysis,\
    #                                            err_style="unit_traces") # uncomment for unit traces
                                                err_style="ci_band", ci = conf_ints) # uncomment for CIs
        sns.plt.title(title)
        sns.plt.ylabel("Correlation")
        sns.plt.ylim([-1, 1])
#        plt.savefig("./correlation_figs/{data_name}/{label}.svg".format(label=analysis, data_name = a_trajectory), format="svg")
        plt.savefig("./correlation_figs/{label}.svg".format(label=analysis), format="svg")
        sns.plt.show()


def main():
    csv_list = make_csv_name_list()      
    print csv_list
    csv_list = ['Right Plume-01', 'Right Plume-02', 'Right Plume-03', 'Right Plume-04', 'Right Plume-05', 'Right Plume-06', 'Right Plume-07']
    
    analysis_DF = loop_csvs(csv_list)

    plt.style.use('ggplot')
    plot_analysis(analysis_DF)
    
    return analysis_DF
    
analysis_DF = main()