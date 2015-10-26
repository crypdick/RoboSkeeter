# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:18:13 2015

@author: richard


TODO meeting
manually plot
fix colors
fix data matrix
"""

import os
from glob import glob
import csv
import time

import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt
import pandas as pd

from scripts import i_o

INTERESTED_VALS = ['velo_x', 'velo_y', 'velo_z', 'curve']

#WINDOW_LEN = int(floor( len(df.index)/ 5 ))
WINDOW_LEN = 100
LAGS = 20
MIN_TRAJECTORY_LEN = 400
CONFINT_THRESH = 0.5

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
        df = i_o.load_csv2DF(csv_fname)
        df_vars = df[INTERESTED_VALS] # slice only cols we want
        df_vars['log_curve'] = np.log(df_vars.loc[:,'curve'])
        df_list.append(df_vars)
    INTERESTED_VALS.append('log_curve')

    return pd.concat(df_list)


def DF2analyzedSegments(DF):
    print "Segmenting data, running ACF + PACF analysis."
#    analyzed_segment_list = []
#    super_data_list = []
#    confint_lower_list = []
#    confint_upper_list = []
    filtered_data_list = []
    t0 = time.time()
    for csv_fname, DF in trajectory_DF.groupby(level=0):
#        pool = mp.Pool(processes = 4)
#        results = [pool.apply(segment_analysis, args=(csv_fname, DF)) for csv_fname, DF in DF_dict.iteritems()]
#        analysis_df, super_data_matrix = segment_analysis(csv_fname, DF)
#        super_data_panel, confint_lower, confint_upper = segment_analysis(csv_fname, DF)
        filtered_panel = segment_analysis(csv_fname, DF)
#        if analysis_df is None: # our df was too small to analyze
#            pass
#        else:
#            analyzed_segment_list.append(analysis_df)
#        super_data_list.append(super_data_panel)
#        confint_lower_list.append(confint_lower)
#        confint_upper_list.append(confint_upper)
        filtered_data_list.append(filtered_panel)
    t1 = time.time()
    print "Segment analysis finished in %f seconds." % (t1-t0)
#    return pd.concat(results)
#    return pd.concat(analyzed_segment_list), super_data_matrix 
#    print super_data_list
    return pd.concat(filtered_data_list, axis=1)
#    return pd.concat(super_data_list, axis=1), pd.concat(confint_lower_list, axis=1), pd.concat(confint_upper_list, axis=1), 
            

def segment_analysis(csv_fname, trajectory_df):
    # catch small trajectory_dfs
    if len(trajectory_df.index) < MIN_TRAJECTORY_LEN:
        return None
    else:
        num_segments = len(trajectory_df.index) - WINDOW_LEN
        
        # for each trajectory, loop through segments

#        super_data = np.zeros((num_segments+1, LAGS+1+1, 2*len(INTERESTED_VALS)+1))
#        super_data = np.zeros((2*len(INTERESTED_VALS), num_segments, LAGS+1))
#        super_data_confint_upper = np.zeros((2*len(INTERESTED_VALS), num_segments, LAGS+1))
#        super_data_confint_lower = np.zeros((2*len(INTERESTED_VALS), num_segments, LAGS+1))
        confident_data = np.zeros((2*len(INTERESTED_VALS), num_segments, LAGS+1))
#        segmentnames = np.ndarray.flatten( np.array([["{name:s} seg{index:0>3d}".format(name="C", index=segment_i)]*(LAGS+1) for segment_i in range(num_segments)]) )
        
        for segment_i in range(num_segments):
            # slice out segment from trajectory
            segment = trajectory_df[segment_i:segment_i+WINDOW_LEN]
            
#            data_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
#            confint_matrix = np.zeros((2*len(INTERESTED_VALS), LAGS+1))
            
            ## for segment, run PACF and ACF for each feature
            
            # do analysis variable by variable
            for var_name, var_values in segment.iteritems():
                # make matrices
                
                
                
                # make dictionary for column indices
                var_index = segment.columns.get_loc(var_name)
#                {'velo_x':0, 'velo_y':1, 'velo_z':2, 'curve':3, 'log_curve':4}[var_name]
                
                # run ACF and PACF for the column
                col_acf, acf_confint = acf(var_values, nlags=LAGS, alpha=.05)#,  qstat= True)
                
                # store data
#                super_data[var_index, segment_i, :] = col_acf
#                super_data_confint_lower[var_index, segment_i, :] = acf_confint[:,0]
#                super_data_confint_upper[var_index, segment_i, :] = acf_confint[:,1]
                # make confident data
                acf_confint_distance = acf_confint[:,1] - acf_confint[:,0]
                ACF_conf_booltable = acf_confint_distance[:] >= CONFINT_THRESH
                filtered_data = col_acf
                filtered_data[ACF_conf_booltable] = 0.
                confident_data[var_index, segment_i, :] = filtered_data
                
                
                
                ## , acf_confint, acf_qstats, acf_pvals
                col_pacf, pacf_confint = pacf(var_values, nlags=LAGS, method='ywmle', alpha=.05)
                # TODO: check for PACF values above or below +-1
#                super_data[var_index+len(INTERESTED_VALS), segment_i, :] = col_pacf
#                super_data_confint_lower[var_index+len(INTERESTED_VALS), segment_i, :] = pacf_confint[:,0]
#                super_data_confint_upper[var_index+len(INTERESTED_VALS), segment_i, :] = pacf_confint[:,1]
                
                # make confident data
                pacf_confint_distance = pacf_confint[:,1] - pacf_confint[:,0]
                PACF_conf_booltable = pacf_confint_distance[:] >= CONFINT_THRESH
                filtered_data = col_pacf # make a copy
                filtered_data[PACF_conf_booltable] = 0.
                confident_data[var_index+len(INTERESTED_VALS), segment_i, :] = filtered_data
                

                
                
            
        # analysis panel  
        major_axis=[np.array([csv_fname]*num_segments), np.array(["{index:0>3d}".format(index=segment_i) for segment_i in range(num_segments)])]
        
#        p = pd.Panel(super_data,
#             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
##            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
#            major_axis=major_axis,            
#            minor_axis=np.arange(LAGS+1))
#        p.major_axis.names = ['Trajectory', 'segment_ID']
#
#        # confint panel
#        p_confint_upper = pd.Panel(super_data_confint_upper,
#             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
##            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
#            major_axis=major_axis,            
#            minor_axis=np.arange(LAGS+1))
#        p_confint_upper.major_axis.names = ['Trajectory', 'segment_ID']  
#        
#        p_confint_lower = pd.Panel(super_data_confint_lower,
#             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
##            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
#            major_axis=major_axis,            
#            minor_axis=np.arange(LAGS+1))
#        p_confint_lower.major_axis.names = ['Trajectory', 'segment_ID'] 
        
        # analysis panel  
        
        filtpanel = pd.Panel(confident_data,
             items=['acf_velox', 'acf_veloy','acf_veloz', 'acf_curve', 'acf_logcurve', 'pacf_velox', 'pacf_veloy', 'pacf_veloz', 'pacf_curve', 'pacf_logcurve'],
#            major_axis=np.array(["{name:s} seg{index:0>3d}".format(name=csv_fname, index=segment_i) for segment_i in range(num_segments)]),
            major_axis=major_axis,            
            minor_axis=np.arange(LAGS+1))
        filtpanel.major_axis.names = ['Trajectory', 'segment_ID']
        
        
        return filtpanel
#        return p, p_confint_upper, p_confint_lower, filtpanel
                  

                      
def plot_analysis(analysis_panel):#, confint_lower_panel, confint_upper_panel):
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
#        DF_lower = confint_lower_panel[analysis].sortlevel(0)
#        DF_upper = confint_upper_panel[analysis].sortlevel(0)
        #TODO figure out DF.index.lexsort_depth error
        for csv_fname, df in DF.groupby(level=0):
            if not os.path.exists('./correlation_figs/{data_name}'.format(data_name = csv_fname)):
                os.makedirs('./correlation_figs/{data_name}'.format(data_name = csv_fname))
            
            # num segs in this csv
            num_segs = df.shape[0] *1.0 # turn to floats
            
            # select confint data
#            df_lower = DF_lower.xs(csv_fname, level='Trajectory')
#            df_upper = DF_upper.xs(csv_fname, level='Trajectory')
            
#            fig = plt.figure()
#            plt.title(csv_fname + " " + title)
#            plt.ylabel("Correlation")
#            plt.xlabel("Lags")
#            plt.ylim([-1, 1])
#            
#            seg_iterator = df.iterrows()
#            
#            # plot flat
#            color = iter(plt.cm.Set2(np.linspace(0,1,num_segs)))
#            for index, seg in seg_iterator:
#                c=next(color)
#                sns.plt.plot(seg, color=c, alpha=0.6)
#            plt.plot(range(21), np.zeros(21), color='lightgray')
#            plt.savefig("./correlation_figs/{data_name}/{data_name} - 2D{label}.svg".format(label=analysis, data_name = csv_fname), format="svg")
            
            # plot as a surface
            surfacefig = plt.figure()
            surfaceax = surfacefig.gca(projection='3d')
            plt.title(csv_fname + " " + title)
            
            x = np.arange(LAGS+1.0)
            y = np.arange(num_segs)
            
                
            
            XX, YY = np.meshgrid(x, y)
    
            surf = surfaceax.plot_surface(XX, YY, df, shade=False,
                             facecolors=plt.cm.Set2((YY-YY.min()) / (YY.max()-YY.min())), cstride=1, rstride=5, alpha=0.7)
            # add grey plane at corr=0
            zeroplane = np.zeros_like(XX)
            surfaceax.plot_surface(XX, YY, zeroplane, color='lightgray', linewidth=0, alpha=0.3)
#            # plot upper conf int
#            surfaceax.plot_surface(XX, YY, df_upper, color='r', alpha=0.1, linewidth=0)
#            surfaceax.plot_surface(XX, YY, df_lower, color='r', alpha=0.1, linewidth=0)
                    
            
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
#analysis_panel, confint_lower_panel, confint_upper_panel = DF2analyzedSegments(trajectory_DF)
filtered_panel = DF2analyzedSegments(trajectory_DF)
##
###plt.style.use('ggplot')
###graph_matrix = plot_analysis(segment_analysis_DF)
#plot_analysis(analysis_panel, confint_lower_panel, confint_upper_panel)
plot_analysis(filtered_panel)
