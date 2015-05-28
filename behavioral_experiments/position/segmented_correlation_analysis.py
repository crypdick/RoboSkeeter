# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:18:13 2015

@author: richard


TODO meeting
manually plot
fix colors
fix data matrix
"""


import trajectory_data_io
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from math import floor as floor
import numpy as np
import seaborn as sns
import pandas as pd
from glob import glob
import csv
import multiprocessing as mp
import time

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
    # TODO export this to io
    dyn_traj_reldir = "data/dynamical_trajectories/"
    print "Loading + filtering CSV files from ", dyn_traj_reldir
    os.chdir(dyn_traj_reldir)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv") if sum(1 for row in csv.reader(open(file))) > MIN_TRAJECTORY_LEN])
#    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__))
    
    return csv_list


def csvList2dfList(csv_list):
    # TODO export this to io
    print "Extracting csv data."
    df_dict = {}
    for csv_fname in csv_list:
        df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
        df_vars = df[INTERESTED_VALS] # slice only cols we want
        df_vars['log_curve'] = np.log(df_vars.loc[:,'curve'])
        INTERESTED_VALS.append('log_curve')
        df_dict[csv_fname] = df_vars
        
    return df_dict


def DF_dict2analyzedSegments(DF_dict):
    print "Segmenting data, running ACF + PACF analysis."
    analyzed_segment_list = []
    super_data_list = []
    t0 = time.time()
    for csv_fname, DF in DF_dict.iteritems():
#        pool = mp.Pool(processes = 4)
#        results = [pool.apply(segment_analysis, args=(csv_fname, DF)) for csv_fname, DF in DF_dict.iteritems()]
#        analysis_df, super_data_matrix = segment_analysis(csv_fname, DF)
        super_data_panel = segment_analysis(csv_fname, DF)
#        if analysis_df is None: # our df was too small to analyze
#            pass
#        else:
#            analyzed_segment_list.append(analysis_df)
        super_data_list.append(super_data_panel)
    t1 = time.time()
    print "Segment analysis finished in %f seconds." % (t1-t0)
#    return pd.concat(results)
#    return pd.concat(analyzed_segment_list), super_data_matrix 
    return super_data_list[0] # TODO get rid of 0
            

def segment_analysis(csv_fname, trajectory_df):
    # catch small trajectory_dfs
    if len(trajectory_df.index) < MIN_TRAJECTORY_LEN:
        return None
    else:
        num_segments = len(trajectory_df.index) - WINDOW_LEN
        
        # for each trajectory, loop through segments
        analysis_df = []
        segment_df_list = {}
#        super_data = np.zeros((num_segments+1, LAGS+1+1, 2*len(INTERESTED_VALS)+1))
        super_data = np.zeros((2*len(INTERESTED_VALS), num_segments, LAGS+1))
#        segmentnames = np.ndarray.flatten( np.array([["{name:s} seg{index:0>3d}".format(name="C", index=segment_i)]*(LAGS+1) for segment_i in range(num_segments)]) )
        
        for segment_i in range(num_segments):
            # slice out segment from trajectory
            segment = trajectory_df[segment_i:segment_i+WINDOW_LEN]
            
#            segmentdict = {}
            data_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
#            ACF_matrix = np.empty((num_segments,  LAGS+1))
#            PACF_matrix = np.empty((num_segments,  LAGS+1))
            
            ## for segment, run PACF and ACF for each feature
            
            # do analysis variable by variable
            for var_name, var_values in segment.iteritems():
                # make matrices
                
                
                
                # make dictionary for column indices
                var_index = segment.columns.get_loc(var_name)
#                {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3, 'log_curve':4}[var_name]
                
                # run ACF and PACF for the column
                col_acf = acf(var_values, nlags=LAGS)#, alpha=.05,  qstat= True)
                
                # store data
                data_matrix[var_index] = col_acf
                super_data[var_index, segment_i, :] = col_acf
#                segmentdict['ACF'+var_name] = col_acf
                
                
                ## , acf_confint, acf_qstats, acf_pvals
                col_pacf = pacf(var_values, nlags=LAGS, method='ywmle')
                # TODO: check for PACF values above or below +-1
                data_matrix[var_index+len(INTERESTED_VALS)] = col_pacf
                super_data[var_index+len(INTERESTED_VALS), segment_i, :] = col_pacf
#                print "type2 ", type(super_data)
#                segmentdict['PACF'+var_name] = col_pacf
                
                
            
            
            
#            # turn data matrix for our segment into a pd and add it to the list
#            analysis_df.append(pd.DataFrame(dict(#Condition=[["control", "left_heater", "right_heater"][0]] * (LAGS+1),
#                                 Condition=[csv_fname] * (LAGS+1),
#                                            Segment=["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i)] * (LAGS+1),
#                                            Lags=range(LAGS+1),
#                                            acf_velox=data_matrix[0],
#                                            acf_veloy=data_matrix[1],
#                                            acf_veloz=data_matrix[2],
#                                            acf_curve=data_matrix[3],
#                                            acf_logcurve=data_matrix[4],
#                                            pacf_velox=data_matrix[0+len(INTERESTED_VALS)],
#                                            pacf_veloy=data_matrix[1+len(INTERESTED_VALS)],
#                                            pacf_veloz=data_matrix[2+len(INTERESTED_VALS)],
#                                            pacf_curve=data_matrix[3+len(INTERESTED_VALS)],
#                                            pacf_logcurve=data_matrix[4+len(INTERESTED_VALS)]
#                                            )))
            
#        
        p = pd.Panel(super_data,
             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
            minor_axis=np.arange(LAGS+1))
#        return pd.concat(analysis_df), p
        return p
                  

                      
def plot_analysis(analysis_DF):
    print "Plotting."
    analysis_types = {'acf_velox': "ACF Velocity x", 'acf_veloy': "ACF Velocity y",\
                 'acf_veloz': "ACF Velocity Z", 'acf_curve': "ACF curvature", \
                 'acf_logcurve': "ACF log(curvature)",\
                'pacf_velox': "PACF Velocity x", 'pacf_veloy': "PACF Velocity y", \
                'pacf_veloz': "PACF Velocity z", 'pacf_curve': "PACF curvature", \
                'pacf_logcurve': "PACF log(curvature)"}
    
    
    
#    unique_seg_names = np.unique(analysis_DF.Segment.values)
    num_segs = analysis_DF.shape[1]
    
    
#    graph_matrix = np.empty((len(unique_seg_names), LAGS+1))
#    graph_matrix.fill(np.nan)
    # "for each unique segment/segment index..."
#    for i in range(num_segs):
#        # ... look up seg name, grab the ACF values
#        segment_slice = analysis_DF.loc[segment_analysis_DF['Segment']==np.unique(analysis_DF.Segment.values)[i],:]
#        segment_ACF = segment_slice.acf_velox
#        graph_matrix[i] = segment_ACF

    for analysis, title in analysis_types.iteritems():
        df = analysis_DF[analysis]
        
        
        fig = plt.figure()
        plt.title(title)
        plt.ylabel("Correlation")
        plt.xlabel("Lags")
        plt.ylim([-1, 1])
        
        seg_iterator = df.iterrows()
        
        # plot flat
        color = iter(plt.cm.Set2(np.linspace(0,1,num_segs)))
        for index, seg in seg_iterator:
            c=next(color)
            sns.plt.plot(seg, color=c)
        
        # plot as a surface
        surfacefig = plt.figure()
        surfaceax = surfacefig.gca(projection='3d')
        x = np.arange(LAGS+1)
        y = np.arange(num_segs)
        
        X, Y = np.meshgrid(x, y)
#        print "x shape", X.shape # 602, 21
#        print "y shape", Y.shape
        
#        color = plt.cm.Set2(np.linspace(0,1,num_segs))
#        color = np.expand_dims(color, axis=1)
#        color = np.tile(color, (1,LAGS+1,1))
#        print color
        surfaceax.plot_surface(X, Y, df,  cmap=plt.cm.Set2(fc))#, color=color)
        plt.show()
        
        
#        fig = plt.figure()
#        conf_ints = [95, 68]
#        sns.tsplot(df)
##        sns.tsplot(df, time="Lags", unit="Segment", condition="Condition",\
##                     value=analysis, err_palette= palette,\
##                                                err_style="unit_traces") # uncomment for unit traces
###                                                err_style="ci_band", ci = conf_ints) # uncomment for CIs
#        sns.plt.title(title)
#        sns.plt.ylabel("Correlation")
#        sns.plt.ylim([-1, 1])
##        plt.savefig("./correlation_figs/{data_name}/{label}.svg".format(label=analysis, data_name = a_trajectory), format="svg")
#        plt.savefig("./correlation_figs/{label}.svg".format(label=analysis), format="svg")
#        sns.plt.show()

    
    
    
#    if np.isnan(graph_matrix).any(): # we have nans
#        print "Error! NaNs in matrix!"

    
#    return graph_matrix
#    sweet stuff ######
#    df.mean(axis=0)
#    df.var(axis=0)
#    df.std(axis=0)
    



    
    
#csv_list = make_csv_name_list()

csv_list = ['Control-27']
#csv_list = ['Right Plume-01', 'Right Plume-02', 'Right Plume-03', 'Right Plume-04', 'Right Plume-05']#, 'Right Plume-06', 'Right Plume-07']
trajectory_DFs_dict = csvList2dfList(csv_list)
trajectory_DF = pd.concat(trajectory_DFs_dict.values())

#segment_analysis_DF, super_data = DF_dict2analyzedSegments(trajectory_DFs_dict)
super_data = DF_dict2analyzedSegments(trajectory_DFs_dict)

#plt.style.use('ggplot')
#graph_matrix = plot_analysis(segment_analysis_DF)
graph_matrix = plot_analysis(super_data)
