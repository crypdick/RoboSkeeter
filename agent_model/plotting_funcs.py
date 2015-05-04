# -*- coding: utf-8 -*-
"""
Flight trajectory plotting functions

Created on Fri Mar 20 12:27:04 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_palette("muted", 8)


def plot_single_trajectory(dynamics, metadata, plot_kwargs=None):
    # plot an individual trajectory
    try:
        plt.plot(dynamics['position_x'], dynamics['position_y'], lw=2, alpha=0.5)
    except KeyError:
        print dynamics.keys()
        pass
    currentAxis = plt.gca()
    cage = draw_cage()
    currentAxis.add_patch(cage)
    currentAxis.axis([0,1,0.127, -0.127])
    if metadata['target_position'] is not None:
        heaterCircle, detectCircle = draw_heaters(metadata['target_position'], metadata['detection_threshold'])
        currentAxis.add_artist(heaterCircle)
        currentAxis.add_artist(detectCircle)
    plt.gca().set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(15, 4.5)
    
    plt.title(plot_kwargs['title'] + plot_kwargs['titleappend'], fontsize=20)
    plt.xlabel("Upwind/$x$ (meters)", fontsize=14)
    plt.ylabel("Crosswind/$y$ (meters)", fontsize=14)
    plt.savefig("./figs/indiv_traj beta{beta}_f{rf}_wf{wtf}_bounce {bounce}.svg".format(beta=metadata['beta'], rf=metadata['rf'], wtf=metadata['wtf'], bounce=metadata['bounce']), format="svg")    


def draw_cage():
    cage_midX, cage_midY = 0.1524, 0.
    return plt.Rectangle((cage_midX - 0.0381, cage_midY - 0.0381), 0.0762, 0.0762, facecolor='none')
    

def draw_heaters(target_pos, detect_thresh):
    heaterCircle = plt.Circle((target_pos[0], target_pos[1],), 0.003175, color='r')  # 0.003175 is diam of our heater
    detectCircle = plt.Circle((target_pos[0], target_pos[1],), detect_thresh, color='gray', fill=False, linestyle='dashed')
    return heaterCircle, detectCircle


def trajectory_plots(ensemble, metadata, plot_kwargs=None):
    """"Plot all the trajectories into a single arena"""
    fig, ax = plt.subplots(1)
#    sns.set_style("white")
#    for trajectory_i in range(metadata['total_trajectories']):
#        posx = ensemble.xs(trajectory_i, level='trajectory')['position_x']
##        if metadata['total_trajectories'] < 60:
##            alpha=0.4
##        else:
##            alpha=0.02
#        if trajectoryPlot is True:
#            posy = ensemble.xs(trajectory_i, level='trajectory')['position_y']
#            ax.plot(posx, posy, lw=2, alpha=1)
#        ax.axis([0,1,0.127, -0.127])  # slight y padding for graphs
#    title_append = r""" $T_max$ = {0} secs, $\beta = {2}$, $f = {3}$, $wtf = {4}$.
##                """.format(metadata['time_max'], metadata['total_trajectories'], metadata['beta'], metadata['rf'], metadata['wtf'])
#                

#
##    
#    # plot shwag
#    title_append = title_append + """<Tfind> = {0:}, Sourcefinds = {1}/(n = {2})""".format( metadata['time_target_find_avg'],metadata['total_finds'], metadata['total_trajectories'])
#    plt.title("Agent trajectories" + title_append, fontsize=14)
#    plt.xlabel("$x$", fontsize=14)
#    plt.ylabel("$y$", fontsize=14)
#    plt.axis(metadata['boundary'])
#    sns.set_style("white")
##    
#    # save before overlaying heatmap
#    plt.savefig("./figs/Trajectories b{beta} f{rf} wf{wtf} bounce {bounce} N{total_trajectories}.png"\
#        .format(beta=metadata['beta'], rf=metadata['rf'], wtf=metadata['wtf'], bounce=metadata['bounce'], total_trajectories=metadata['total_trajectories']))
#    
############################################## NEW HEATMAP #######################
#    ## Position heatmap
    if plot_kwargs['heatmap']:
#        with sns.axes_style("white"):
#            hextraj = sns.jointplot('position_x', 'position_y', ensemble, size=10)#, ylim=(-.15, .15))
#            hextraj.plot_marginals(sns.distplot, kde=False)
#            hextraj.plot_joint(plt.hexbin, vmax=30, extent = [metadata['boundary'][0], metadata['boundary'][1], metadata['boundary'][3], metadata['boundary'][2]])#joint_kws={'gridsize':(100,30), })
#            hextraj.ax_joint.set_aspect('equal')
#            hextraj.ax_joint.invert_yaxis()  # hack to match y axis convention 
#            cax = hextraj.fig.add_axes([1, .25, .04, .5])
#            plt.colorbar(cax=cax)
#            
#            if metadata['target_position'] is not None:
#                heaterCircle, detectCircle = draw_heaters(metadata['target_position'], metadata['detection_threshold'])
#                hextraj.ax_joint.add_artist(heaterCircle)
#                hextraj.ax_joint.add_artist(detectCircle)
#        #    
#            # draw cage
#            cage = draw_cage()
#            hextraj.ax_joint.add_patch(cage)
#        plt.savefig("./figs/Trajectories sns heatmap beta{beta}_f{rf}_wf{wtf}_bounce {bounce} N{total_trajectories}.png".format(beta=metadata['beta'], rf=metadata['rf'], wtf=metadata['wtf'], bounce=metadata['bounce'], total_trajectories=metadata['total_trajectories']))
#            
        ########## OLD HEATMAP ##############################################
        # crunch the data
#        counts, xedges, yedges = np.histogram2d(ensemble['position_x'], ensemble['position_y'], bins=(100,30), range=[[0, 1], [-0.15, .15]])
        # only considering trajectories between 0.25 - 0.85 m in x direction   
        
        trim_ensemble = ensemble.loc[(ensemble['position_x']>0.25) & (ensemble['position_x']<0.85)]
        counts, xedges, yedges = np.histogram2d(trim_ensemble['position_x'], trim_ensemble['position_y'], bins=(100,30), range=[[0.25, 0.85], [-0.127, .127]])
        
        
        
        # counts needs to be transposed to use pcolormesh     
        counts = counts.T
        probs = counts/ metadata['total_trajectories']
        
#        MaxVal = metadata['total_trajectories']/2
#        if metadata['total_trajectories'] > 100:
#            plt.cla()
        heatmap = ax.pcolormesh(xedges, yedges, probs, cmap=plt.cm.Oranges, vmin=0., vmax=.2)
        if metadata['target_position'] is not None:
                heaterCircle, detectCircle = draw_heaters(metadata['target_position'], metadata['detection_threshold'])
                ax.add_artist(heaterCircle)
                ax.add_artist(detectCircle)
#        #    
#            # draw cage
#            cage = draw_cage()
#            hextraj.ax_joint.add_patch(cage)
        ax.set_aspect('equal')
#        ax.invert_yaxis()  # hack to match y axis convention --- now unneeded?
        ax.set_ylim([0.127, -.127])  # keep positive first to invert axis
        ax.set_xlim([0.25, 0.85])
        
        # overwrite previous plot schwag
        cbar = plt.colorbar(heatmap, shrink=0.5, pad=0.05)
        cbar.ax.set_ylabel('Probability')
        plt.title("Agent trajectories 2D position histogram (n = {})".format(metadata['total_trajectories']))
        plt.xlabel("Upwind/$x$ (meters)")
        plt.ylabel("Crosswind/$y$ (meters)")
        plt.savefig("./figs/Trajectories heatmap beta{beta}_f{rf}_wf{wtf}_bounce {bounce} N{total_trajectories}.svg".format(beta=metadata['beta'], rf=metadata['rf'], wtf=metadata['wtf'], bounce=metadata['bounce'], total_trajectories=metadata['total_trajectories']), format="svg")
        plt.show()
        #####################################################################


def stateHistograms(ensemble, metadata, plot_kwargs=None):
    trim_ensemble = ensemble.loc[(ensemble['position_x']>0.25) & (ensemble['position_x']<0.85)]
    
    fig = plt.figure(4, figsize=(9, 8))
    gs1 = gridspec.GridSpec(2, 2)
    axs = [fig.add_subplot(ss) for ss in gs1]
    fig.suptitle("Agent Model Flight Distributions", fontsize=14)   
    # position distributions
#    pos_all = np.concatenate(pos, axis=0)
    pos_binwidth = .01
    
    # X pos
    xpos_min, xpos_max = 0.25, .85
    xpos_counts, xpos_bins = np.histogram(trim_ensemble['position_x'], bins=np.linspace(xpos_min, xpos_max, (xpos_max-xpos_min) / pos_binwidth))
    xpos_counts_n = xpos_counts.astype(int) / float(xpos_counts.size)
    axs[0].plot(xpos_bins[:-1]+pos_binwidth/2, xpos_counts_n, lw=2)
    axs[0].bar(xpos_bins[:-1], xpos_counts_n, xpos_bins[1]-xpos_bins[0], facecolor='blue', linewidth=0, alpha=0.1)
    axs[0].set_title("Upwind ($x$) Position Distributions", fontsize=12)
    axs[0].set_xlim(xpos_min+pos_binwidth/2, xpos_max-pos_binwidth/2)
    axs[0].set_xlabel("Position ($m$)")
    axs[0].set_ylabel("Probability")
    
    # Y pos
    ypos_min, ypos_max = -0.127, 0.127
    ypos_counts, ypos_bins = np.histogram(trim_ensemble['position_y'], bins=np.linspace(ypos_min, ypos_max, (ypos_max-ypos_min)/pos_binwidth))
    ypos_counts_n = ypos_counts/ ypos_counts.astype(float).sum()
    axs[1].plot(ypos_bins[:-1]+pos_binwidth/2, ypos_counts_n, lw=2)
    axs[1].set_xlim(ypos_min+pos_binwidth/2, ypos_max-pos_binwidth/2)  # hack to hide gaps
    axs[1].fill_between(ypos_bins[:-1]+pos_binwidth/2, 0, ypos_counts_n, facecolor='blue', alpha=0.1)
    axs[1].set_title("Cross-wind ($y$) Position Distributions")
    axs[1].set_xlabel("Position ($m$)")
    axs[1].set_ylabel("Probability")
    

    ## Velo distributions
    vmin, vmax = -1.0, 1.
    velo_bindwidth = 0.02
    
    # vx component
    vx_counts, vx_bins = np.histogram(trim_ensemble['velocity_x'], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    vx_counts_n = vx_counts / vx_counts.astype(float).sum()
    axs[2].plot(vx_bins[:-1], vx_counts_n, label="$\dot{x}$")
    axs[2].fill_between(vx_bins[:-1], 0, vx_counts_n, facecolor='blue', alpha=0.1)
    # vy component
    vy_counts, vy_bins = np.histogram(trim_ensemble['velocity_y'], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    vy_counts_n= vy_counts / vy_counts.astype(float).sum()
    axs[2].plot(vy_bins[:-1], vy_counts_n, label="$\dot{y}$")
    axs[2].fill_between(vy_bins[:-1], 0, vy_counts_n, facecolor='green', alpha=0.1)
    # |v|
    velo_all_magn = []
#    for v in velo_all:
#        velo_all_magn.append(np.linalg.norm(v))
#    vabs_counts, vabs_bins = np.histogram(velo_all_magn, bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
#    vabs_counts_n = vabs_counts / vabs_counts.astype(float).sum()
#    axs[2].plot(vabs_bins[:-1], vabs_counts_n, label='$|\mathbf{v}|$', color=sns.desaturate("black", .4), lw=2)
#    axs[2].fill_between(vabs_bins[:-1], 0, vabs_counts_n, facecolor='yellow', alpha=0.1)
    
    axs[2].set_title("Velocity Distributions")
    axs[2].set_xlabel("Velocity ($m/s$)", fontsize=12)
    axs[2].set_ylabel("Probability", fontsize=12)
    axs[2].legend(fontsize=14)
    
    ## Acceleration distributions
#    accel_all = np.concatenate(accels, axis=0)
    
    amin, amax = -10., 10
    accel_binwidth = 0.2
    
    # ax component
    ax_counts, ax_bins = np.histogram(trim_ensemble['acceleration_x'], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    ax_counts_n = ax_counts / ax_counts.astype(float).sum()
    axs[3].plot(ax_bins[:-1], ax_counts_n, label="$\ddot{x}$", lw=2)
    axs[3].fill_between(ax_bins[:-1], 0, ax_counts_n, facecolor='blue', alpha=0.1)
    # ay component
    ay_counts, ay_bins = np.histogram(trim_ensemble['acceleration_y'], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    ay_counts_n = ay_counts / ay_counts.astype(float).sum()
    axs[3].plot(ay_bins[:-1], ay_counts_n, label="$\ddot{y}$", lw=2)
    axs[3].fill_between(ay_bins[:-1], 0, ay_counts_n, facecolor='green', alpha=0.1)
    # |a|
#    accel_all_magn = []
#    for a in accel_all:
#        accel_all_magn.append(np.linalg.norm(a))
#    aabs_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
#    aabs_counts_n = aabs_counts/ aabs_counts.astype(float).sum()
#    axs[3].plot(aabs_bins[:-1], aabs_counts_n, label='$|\mathbf{a}|$', color=sns.desaturate("black", .4), lw=2)
#    axs[3].fill_between(aabs_bins[:-1], 0, aabs_counts_n, facecolor='yellow', alpha=0.1)
    axs[3].set_title("Acceleration Distribution")
    axs[3].set_xlabel("Acceleration ($m^s/s$)")
    axs[3].set_ylabel("Probability")
    axs[3].legend(fontsize=14)
    
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])  # overlapping text hack
    plt.savefig("./figs/Agent Distributions b {beta},f {rf},wf {wtf},bounce {bounce},N {total_trajectories}.svg".format(beta=metadata['beta'], rf=metadata['rf'], wtf=metadata['wtf'], bounce=metadata['bounce'], total_trajectories=metadata['total_trajectories']), format='svg')
    
    
    return xpos_counts_n, ypos_bins, ypos_counts, ypos_counts_n, vx_counts_n


def force_scatter(ensemble):
    
    trim_ensembleF = ensemble.loc[(ensemble['position_x']>0.25) & (ensemble['position_x']<0.85), ['totalF_x', 'totalF_y', 'randF_x', 'randF_y', 'stimF_x', 'stimF_y', 'upwindF_x', 'upwindF_y', 'wallRepulsiveF_x', 'wallRepulsiveF_y']]
    # plot Forces
#    f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
##    forcefig = plt.figure(5, figsize=(9, 8))
##    gs2 = gridspec.GridSpec(2, 2)
##    Faxs = [fig.add_subplot(ss) for ss in gs2]
#    forcefig = plt.figure(5)
#    Faxs1 = forcefig.add_subplot(211)
#    Faxs2 = forcefig.add_subplot(212)
    sns.violinplot(trim_ensembleF, color="Paired", lw=2, alpha=0.7)
#    tF = sns.jointplot('totalF_x', 'totalF_y', trim_ensemble, kind="hex", size=10)
    plt.suptitle("Force distributions")
    plt.tight_layout(pad=1.3)    

    plt.ylabel("Force magnitude distribution")
    



if __name__ == '__main__':
    import trajectory_stats
    
        # wallF params
    wallF_max=5e-7
    decay_const = 250
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 5e-7  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)
    
    ensemble, metadata = trajectory_stats.main()
        
    trajectory_plots(ensemble, metadata, heatmap=True, trajectoryPlot = True)
    xpos_counts_n, ypos_bins, ypos_counts, ypos_counts_n, vx_counts_n = stateHistograms(ensemble, metadata)