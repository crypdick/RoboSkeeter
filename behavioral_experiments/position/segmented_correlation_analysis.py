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


def csvList2df(csv_list):
    # TODO export this to io
    print "Extracting csv data."
    df_list = []
    for csv_fname in csv_list:
        df = trajectory_data_io.load_trajectory_dynamics_csv(csv_fname)
        df_vars = df[INTERESTED_VALS] # slice only cols we want
        df_vars['log_curve'] = np.log(df_vars.loc[:,'curve'])
        df_list.append(df_vars)
    INTERESTED_VALS.append('log_curve')

    return pd.concat(df_list)


def DF2analyzedSegments(DF):
    print "Segmenting data, running ACF + PACF analysis."
    analyzed_segment_list = []
    super_data_list = []
    t0 = time.time()
    for csv_fname, DF in trajectory_DF.groupby(level=0):
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
#    print super_data_list
    return pd.concat(super_data_list, axis=1)
            

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
            
            data_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
            
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
            
        major_axis=[np.array([csv_fname]*num_segments), np.array(["{index:0>3d}".format(index=segment_i) for segment_i in range(num_segments)])]
        p = pd.Panel(super_data,
             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
#            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
            major_axis=major_axis,            
            minor_axis=np.arange(LAGS+1))
        p.major_axis.names = ['Trajectory', 'segment_ID']
#        return pd.concat(analysis_df), p
        return p
                  

                      
def plot_analysis(analysis_panel):
    print "Plotting."
    analysis_types = {'acf_velox': "ACF Velocity x", 'acf_veloy': "ACF Velocity y",\
                 'acf_veloz': "ACF Velocity Z", 'acf_curve': "ACF curvature", \
                 'acf_logcurve': "ACF log(curvature)",\
                'pacf_velox': "PACF Velocity x", 'pacf_veloy': "PACF Velocity y", \
                'pacf_veloz': "PACF Velocity z", 'pacf_curve': "PACF curvature", \
                'pacf_logcurve': "PACF log(curvature)"}
    type2raw = {'acf_velox': "velo_x", 'acf_veloy': "velo_y",\
                 'acf_veloz': "velo_z", 'acf_curve': "curve", \
                 'acf_logcurve': "log_curve",\
                'pacf_velox': "velo_x", 'pacf_veloy': "velo_y", \
                'pacf_veloz': "velo_z", 'pacf_curve': "curve", \
                'pacf_logcurve': "log_curve"}
    
#    print type(analysis_panel)
   
    

    for analysis, title in analysis_types.iteritems():
        DF = analysis_panel[analysis].sortlevel(0)
        #TODO figure out DF.index.lexsort_depth error
        for csv_fname, df in DF.groupby(level=0):
            if not os.path.exists('./correlation_figs/{data_name}'.format(data_name = csv_fname)):
                os.makedirs('./correlation_figs/{data_name}'.format(data_name = csv_fname))
            
            # num segs in this csv
            num_segs = df.shape[0] *1.0 # turn to floats
            
            
            fig = plt.figure()
            plt.title(csv_fname + " " + title)
            plt.ylabel("Correlation")
            plt.xlabel("Lags")
            plt.ylim([-1, 1])
            
            seg_iterator = df.iterrows()
            
            # plot flat
            color = iter(plt.cm.Set2(np.linspace(0,1,num_segs)))
            for index, seg in seg_iterator:
                c=next(color)
                sns.plt.plot(seg, color=c, alpha=0.6)
            plt.plot(range(21), np.zeros(21), color='lightgray')
            plt.savefig("./correlation_figs/{data_name}/{data_name} - 2D{label}.svg".format(label=analysis, data_name = csv_fname), format="svg")
            
            # plot as a surface
            surfacefig = plt.figure()
            surfaceax = surfacefig.gca(projection='3d')
            x = np.arange(LAGS+1.0)
            y = np.arange(num_segs)
            
                
            
            XX, YY = np.meshgrid(x, y)
#            print csv_fname, "xx", XX.shape, "yy", YY.shape, "df", df.shape
    
            surf = surfaceax.plot_surface(XX, YY, df, shade=False,
                             facecolors=plt.cm.Set2((YY-YY.min())/(YY.max()-YY.min())), cstride=1, rstride=5, alpha=0.7)
            zeroplane = np.zeros_like(XX)
#            print csv_fname, "xx", XX.shape, "yy", YY.shape, "z", zeroplane.shape
            surfaceax.plot_surface(XX, YY, zeroplane, color='lightgray', linewidth=0, alpha=0.3)
                    
            plt.title(csv_fname + " " + title)
            surfaceax.set_xlabel("Lags")
            surfaceax.set_ylabel("Segment Index")
            surfaceax.set_zlabel("Correlation")
            surfaceax.set_zlim(-1, 1)
            plt.draw() # you need this to get the edge color
            line = np.array(surf.get_edgecolor())
            surf.set_edgecolor(line*np.array([0,0,0,0])+1) # make lines white, and keep alpha==1. It's an array of colors like this: [r,g,b,alpha]
            plt.savefig("./correlation_figs/{data_name}/{data_name} - 3D{label}.svg".format(label=analysis, data_name = csv_fname), format="svg")
            
#            # plot relevant raw data, colorized
            variable = type2raw[analysis]
            raw_data = trajectory_DF.xs(csv_fname, level='Trajectory')[variable].values
            x = range(len(raw_data))
            variable_trace = plt.figure()
            ax1 = variable_trace.add_subplot(111) # regular resolution color map
#            
#            
            cm = plt.get_cmap('Set2')
            
            # first we substract WINDOWLEN from range so that we only color the starting
            #points of each window. then we append black values to the end of 
            # the color cycle to make that part of the plots black
            color_cycle = [cm(1.*i/(num_segs-1-WINDOW_LEN)) for i in range(int(num_segs)-1-WINDOW_LEN)]
            color_cycle = color_cycle + [(0., 0., 0., 1.)]*WINDOW_LEN
            ax1.set_color_cycle(color_cycle)
            for i in range(int(num_segs)-1):
                ax1.plot(x[i:i+2], raw_data[i:i+2])
            plt.title(csv_fname + " " + variable)
            plt.xlabel('Trajectory data timestep (ms)')
            plt.ylabel('Value recorded (SI units)')
            plt.savefig("./correlation_figs/{data_name}/{data_name} - raw {variable}.svg".format(data_name = csv_fname, variable = variable), format="svg")
            
                
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
            
    #        plt.savefig("./correlation_figs/{label}.svg".format(label=analysis), format="svg")
    #        sns.plt.show()

    
    
    
#    if np.isnan(graph_matrix).any(): # we have nans
#        print "Error! NaNs in matrix!"

    
#    return graph_matrix
#    sweet stuff ######
#    df.mean(axis=0)
#    df.var(axis=0)
#    df.std(axis=0)
    



    
    
csv_list = make_csv_name_list()

##csv_list = ['Right Plume-39', 'Control-27']
##name = csv_list[0]
##csv_list = ['Right Plume-01', 'Right Plume-02', 'Right Plume-03', 'Right Plume-04', 'Right Plume-05']#, 'Right Plume-06', 'Right Plume-07']
trajectory_DF = csvList2df(csv_list)
#trajectory_DF = pd.concat(trajectory_DFs_dict.values())
#
##segment_analysis_DF, super_data = DF_dict2analyzedSegments(trajectory_DFs_dict)
analysis_panel = DF2analyzedSegments(trajectory_DF)
##
###plt.style.use('ggplot')
###graph_matrix = plot_analysis(segment_analysis_DF)
plot_analysis(analysis_panel)