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
from glob import glob
import csv


INTERESTED_VALS = ['velo_x', 'velo_y', 'velo_z', 'curve']

#WINDOW_LEN = int(floor( len(df.index)/ 5 ))
WINDOW_LEN = 100
LAGS = 20
MIN_TRAJECTORY_LEN = 400

#
## testing parrallelizing code
#def easy_parallize(f, sequence):
#    # I didn't see gains with .dummy; you might
#    from multiprocessing import Pool
#    pool = Pool(processes=8)
#    #from multiprocessing.dummy import Pool
#    #pool = Pool(16)
#
#    # f is given sequence. guaranteed to be in order
#    result = pool.map(f, sequence)
#    cleaned = [x for x in result if not x is None]
#    cleaned = asarray(cleaned)
#    # not optimal but safe
#    pool.close()
#    pool.join()
#    return cleaned


# make list of all csvs in dir
def make_csv_name_list():
    dyn_traj_reldir = "data/dynamical_trajectories/"
    print "Loading CSV files from ", dyn_traj_reldir
    os.chdir(dyn_traj_reldir)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv") if sum(1 for row in csv.reader(open(file))) > MIN_TRAJECTORY_LEN])
#    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__)) 
    
    return csv_list


def csvList2dfList(csv_list):
    df_dict = {}
    for csv_fname in csv_list:
        df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
        df = df[INTERESTED_VALS] # slice only cols we want
        df_dict[csv_fname] = df
    
    return df_dict


def DF_dict2analyzedSegments(DF_dict):
    analyzed_segment_list = []
    
    for csv_fname, DF in DF_dict.iteritems():
        analysis_df = segment_analysis(csv_fname, DF)
        if analysis_df is None: # our df was too small to analyze
            pass
        else:
            analyzed_segment_list.append(analysis_df)
    
    return pd.concat(analyzed_segment_list)
            

def segment_analysis(csv_fname, df):
    # catch small dfs
    if len(df.index) < MIN_TRAJECTORY_LEN:
        return None
    else:
        num_segments = len(df.index) - WINDOW_LEN
        
        # for each trajectory, loop through segments
        analysis_df = []
        for segment_i in range(num_segments):
            segment = df[segment_i:segment_i+WINDOW_LEN]
            
            # for each segment, run PACF and ACF for each feature
            data_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
            
            # calculate ACF and PACF for all our interesting columns
            for label, col in segment.iteritems():
                colindex = {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3}[label]
                
        #        try:
                col_acf = acf(col, nlags=LAGS)#, alpha=.05,  qstat= True)
                
                # store data
                data_matrix[colindex] = col_acf
                ## , acf_confint, acf_qstats, acf_pvals
                col_pacf = pacf(col, nlags=LAGS)
                data_matrix[colindex+len(INTERESTED_VALS)] = col_pacf
            
            # turn data matrix for our segment into a pd and add it to the list
            analysis_df.append(pd.DataFrame(dict(#Condition=[["control", "left_heater", "right_heater"][0]] * (LAGS+1),
                                 Condition=[csv_fname] * (LAGS+1),
                                            Segment=["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i)] * (LAGS+1),
                                            Lags=range(LAGS+1),
                                            acf_velox=data_matrix[0],
                                            acf_veloy=data_matrix[1],
                                            acf_veloz=data_matrix[2],
                                            acf_curve=data_matrix[3],
                                            pacf_velox=data_matrix[4],
                                            pacf_veloy=data_matrix[5],
                                            pacf_veloz=data_matrix[6],
                                            pacf_curve=data_matrix[7]
                                            )))
        return pd.concat(analysis_df)
                  

                      
def plot_analysis(analysis_DF):
    analysis_types = {'acf_velox': "ACF Velocity x", 'acf_veloy': "ACF Velocity y",\
                 'acf_veloz': "ACF Velocity Z", 'acf_curve': "ACF curvature", \
                'pacf_velox': "PACF Velocity x", 'pacf_veloy': "PACF Velocity y", \
                'pacf_veloz': "PACF Velocity z", 'pacf_curve': "PACF curvature"}
                
    for analysis, title in analysis_types.iteritems():
        conf_ints = [95, 68]
        sns.tsplot(analysis_DF, time="Lags", unit="Segment", condition="Condition",\
                     value=analysis, color= sns.color_palette("Blues_r"),\
                                                err_style="unit_traces") # uncomment for unit traces
#                                                err_style="ci_band", ci = conf_ints) # uncomment for CIs
        sns.plt.title(title)
        sns.plt.ylabel("Correlation")
        sns.plt.ylim([-1, 1])
#        plt.savefig("./correlation_figs/{data_name}/{label}.svg".format(label=analysis, data_name = a_trajectory), format="svg")
        plt.savefig("./correlation_figs/{label}.svg".format(label=analysis), format="svg")
        sns.plt.show()


    
#csv_list = make_csv_name_list()      
csv_list = ['Right Plume-01', 'Right Plume-02', 'Right Plume-03', 'Right Plume-04', 'Right Plume-05']#, 'Right Plume-06', 'Right Plume-07']
trajectory_DFs_dict = csvList2dfList(csv_list)
#trajectory_DF = pd.concat(trajectory_DFs_dict.values())

segment_analysis_DF = DF_dict2analyzedSegments(trajectory_DFs_dict)

plt.style.use('ggplot')
plot_analysis(segment_analysis_DF)

