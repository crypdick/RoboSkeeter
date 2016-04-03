# -*- coding: utf-8 -*-
"""
Given CSV filename, plot trajectory
Created on Wed May 27 17:03:55 2015

@author: richard
"""

print """for interactive plots, type this into your iPython console: \n
        """
import os
from glob import glob

import matplotlib.pyplot as plt

from roboskeeter.io import i_o


def plot_trajectory_rainbow(xyz, csv_name, WINDOW_LEN):
###    %matplotlib Qt4Agg
#    palette = sns.color_palette("Pastel1", 450)
    

    fig = plt.figure()
    threedee = fig.gca(projection='3d')
    N = len(xyz.index)
    

    cm = plt.get_cmap('Set2')
    color_cycle = [cm(1.*i/(N-1-WINDOW_LEN)) for i in range(int(N)-1-WINDOW_LEN)]
    color_cycle = color_cycle + [(0., 0., 0., 1.)]*(WINDOW_LEN)
    
    color = iter(color_cycle)
    for i in xrange(N-1):
        c = next(color)
    #    threedee.plot(xyz.pos_x, xyz.pos_y, xyz.pos_z) , color=sns.color_palette("husl", 450*i/N))
        threedee.plot(xyz.pos_y[i:i+2].values, xyz.pos_x[i:i+2].values, xyz.pos_z[i:i+2].values, color=c)#palette[450*i/N])
#        print 450*i/N
        
##    threedee.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]*3])
    
    # Note! I set the y axis to be the X data and vice versa
    threedee.set_ylim(0, 1)
    threedee.set_xlim(0.127, -0.127)
    threedee.set_zlim(0, 0.3)
    threedee.set_xlabel('Y')
    threedee.set_ylabel('X')
    threedee.set_zlabel('Z')
    threedee.set_title('Mosquito trajectory')
    plt.savefig("./correlation_figs/{data_name}/{data_name} Trajectory.svg".format(data_name = csv_name), format="svg")
    plt.show()
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot(xyz.pos_x, xyz.pos_y, xyz.pos_z)
    

def make_csv_name_list():
    # TODO export this to io
    dyn_traj_reldir = "data/processed_trajectories/"
    print "Loading + filtering CSV files from ", dyn_traj_reldir
    os.chdir(dyn_traj_reldir)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
#    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__))
    
    return csv_list

if __name__ == '__main__':
    csv_list = make_csv_name_list()
    print "Please select mosquito trajectory to plot."
    for i, name in enumerate(csv_list):
        print "Option ", i, ": ", name
    
    csv_i = int(raw_input("Trajectory of your desires: "))
    csv_name = csv_list[csv_i]
    
    df = i_o.load_csv_to_df(csv_name)
    xyz = df[['pos_x', 'pos_y', 'pos_z']]
    
    plot_trajectory_rainbow(xyz, csv_name, WINDOW_LEN = 100)
    