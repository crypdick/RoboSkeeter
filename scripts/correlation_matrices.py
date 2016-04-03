# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:00:47 2015

@author: richard
"""
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

#import seaborn as sns
import pandas as pd
from scripts import i_o
from sklearn.metrics.pairwise import pairwise_distances


INTERESTED_VALS = ['velo_x', 'velo_y', 'velo_z']

#WINDOW_LEN = int(floor( len(df.index)/ 5 ))
WINDOW_LEN = 100
LAGS = 20
MIN_TRAJECTORY_LEN = 500

def csvList2df(csv_list):
    # TODO export this to io
    print "Extracting csv data."
    df_list = []
    for csv_fname in csv_list:
        df = i_o.load_csv_to_df(csv_fname)
        df_vars = df[INTERESTED_VALS] # slice only cols we want
        
#        df_vars['log_curve'] = np.log(df_vars.loc[:,'curve'])
        df_list.append(df_vars[0:500]) # make all dfs the same size
#    INTERESTED_VALS.append('log_curve')

    return pd.concat(df_list)
    
def analyze_DF(DF):
    print "Running ACF + PACF analysis."
    ACF_data_list = []
    PACF_data_list = []
    
    counter = -1
    for csv_fname, DF in trajectory_DF.groupby(level=0):
#        pool = mp.Pool(processes = 4)
#        results = [pool.apply(segment_analysis, args=(csv_fname, DF)) for csv_fname, DF in DF_dict.iteritems()]
#        analysis_df, super_data_matrix = segment_analysis(csv_fname, DF)
#        num_trajectories = trajectory_DF.index.get_level_values('Trajectory').unique().size
        acf_data, pacf_data = global_analysis(csv_fname, DF)
#        if analysis_df is None: # our df was too small to analyze
#            pass
#        else:
#            analyzed_segment_list.append(analysis_df)
        ACF_data_list.append(acf_data)
        PACF_data_list.append(pacf_data)
#    return pd.concat(results)
#    return pd.concat(analyzed_segment_list), super_data_matrix 
    return np.hstack(ACF_data_list), np.hstack(PACF_data_list)
#    return pd.concat(super_data_DFs, axis=1)
            

def global_analysis(csv_fname, trajectory_df):
    # catch small trajectory_dfs
    if len(trajectory_df.index) < MIN_TRAJECTORY_LEN:
        return None
    else:
        
        # for each trajectory, loop through segments
        acf_data = np.zeros((len(INTERESTED_VALS), 1, LAGS+1))
        pacf_data = np.zeros((len(INTERESTED_VALS), 1, LAGS+1))
        
            
        # do analysis variable by variable
        count = -1
        for var_name, var_values in trajectory_df.iteritems():
            count += 1
            # make matrices
            
            
            
            # make dictionary for column indices
            var_index = trajectory_df.columns.get_loc(var_name)
#                {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3, 'log_curve':4}[var_name]
            
#            # run ACF and PACF for the column
            col_acf, acf_confint = acf(var_values, nlags=LAGS, alpha=.05)#,  qstat= True)
#            
#            # store data
            acf_data[var_index, 0, :] = col_acf
##            super_data_confint_lower[var_index, segment_i, :] = acf_confint[:,0]
##            super_data_confint_upper[var_index, segment_i, :] = acf_confint[:,1]
            
            
#            ## , acf_confint, acf_qstats, acf_pvals
            col_pacf, pacf_confint = pacf(var_values, nlags=LAGS, method='ywmle', alpha=.05)
            pacf_data[var_index, 0, :] = col_pacf
#            # TODO: check for PACF values above or below +-1
#            super_data[var_index+len(INTERESTED_VALS), segment_i, :] = col_pacf
#            super_data_confint_lower[var_index+len(INTERESTED_VALS), segment_i, :] = pacf_confint[:,0]
#            super_data_confint_upper[var_index+len(INTERESTED_VALS), segment_i, :] = pacf_confint[:,1]

                
                
            
        
        return acf_data, pacf_data


TRAJECTORY_DIR = "data/dynamical_trajectories/"
csv_list = i_o.get_csv_name_list(TRAJECTORY_DIR)

##csv_list = ['Right Plume-39', 'Control-27']
##name = csv_list[0]
##csv_list = ['Right Plume-01', 'Right Plume-02', 'Right Plume-03', 'Right Plume-04', 'Right Plume-05']#, 'Right Plume-06', 'Right Plume-07']
trajectory_DF = csvList2df(csv_list)

ACF_data, PACF_data = analyze_DF(trajectory_DF)
ACF_x, ACF_y, ACF_z = np.vsplit(ACF_data, len(INTERESTED_VALS))
ACF_combined = np.dstack((ACF_x, ACF_y, ACF_z))

## manual edit
#title = 'Velocity Z'
#
## Plot ACF data
#matrix2D = ACF_data[0, :, :]
#Y = pairwise_distances(matrix2D, metric='euclidean')
#PACF_im = plt.imshow(Y, cmap=plt.get_cmap('Reds'), vmax = 1.0)
#plt.colorbar(PACF_im, orientation='horizontal')
#
#plt.title(title + ' ACF distance matrix')
#plt.savefig("./correlation_figs//Distance matrix {label} ACF {Nlags}lags.png".format(label=title, Nlags=LAGS), format="png")
#plt.show()
#
## Plot PACF data
#matrix2D = PACF_data[0, :, :]
#Y = pairwise_distances(matrix2D, metric='euclidean')
#PACF_im = plt.imshow(Y, cmap=plt.get_cmap('Reds'), vmax = 1.0)
#plt.colorbar(PACF_im, orientation='horizontal')
#plt.title(title + ' PACF distance matrix')
#plt.savefig("./correlation_figs//Distance matrix {label} PACF {Nlags}lags.png".format(label=title, Nlags=LAGS), format="png")
#plt.show()


# stacked data
title = 'Combined Velocities'
ACF_x, ACF_y, ACF_z = np.vsplit(ACF_data, len(INTERESTED_VALS))
ACF_combined = np.dstack((ACF_x, ACF_y, ACF_z))

PACF_x, PACF_y, PACF_z = np.vsplit(PACF_data, len(INTERESTED_VALS))
PACF_combined = np.dstack((PACF_x, PACF_y, PACF_z))

## Plot combined ACF data
matrix2D = ACF_combined[0, :, :]
Y = pairwise_distances(matrix2D, metric='euclidean')
PACF_im = plt.imshow(Y, cmap=plt.get_cmap('Reds'))#, vmax = 1.0)
plt.colorbar(PACF_im, orientation='horizontal')

plt.title(title + ' ACF distance matrix')
plt.savefig("./correlation_figs//Distance matrix {label} ACF {Nlags}lags.png".format(label=title, Nlags=LAGS), format="png")
plt.show()

# Plot combined PACF data
matrix2D = PACF_combined[0, :, :]
Y = pairwise_distances(matrix2D, metric='euclidean')
PACF_im = plt.imshow(Y, cmap=plt.get_cmap('Reds'))#, vmax = 1.0)
plt.colorbar(PACF_im, orientation='horizontal')
plt.title(title + ' PACF distance matrix')
plt.savefig("./correlation_figs//Distance matrix {label} PACF {Nlags}lags.png".format(label=title, Nlags=LAGS), format="png")
plt.show()

