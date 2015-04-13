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
import seaborn as sns

sns.set_palette("muted", 8)

def plot_single_trajectory(positionList, target_pos, detect_thresh, boundary, title="Individual trajectory", titleappend=""):
    # plot an individual trajectory
    plt.plot(positionList[:, 0], positionList[:, 1], lw=2, alpha=0.5)
    plt.axis(boundary)
    currentAxis = plt.gca()
    cage = draw_cage()
    currentAxis.add_patch(cage)
    if target_pos is not None:
        heaterCircle, detectCircle = draw_heaters(target_pos, detect_thresh)
        currentAxis.add_artist(heaterCircle)
        currentAxis.add_artist(detectCircle)
    
    plt.title(title + titleappend, fontsize=20)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)


def draw_cage():
    cage_midX, cage_midY = 0.1524, 0.
    return plt.Rectangle((cage_midX - 0.0381, cage_midY - 0.0381), 0.0762, 0.0762, facecolor='none')
    

def draw_heaters(target_pos, detect_thresh):
    heaterCircle = plt.Circle((target_pos[0], target_pos[1],), 0.003175, color='r')  # 0.003175 is diam of our heater
    detectCircle = plt.Circle((target_pos[0], target_pos[1],), detect_thresh, color='gray', fill=False, linestyle='dashed')
    return heaterCircle, detectCircle


def trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list, heatmap):
    """"Plot all the trajectories into a single arena"""
    traj_ex = trajectory_objects_list[0]
    target_pos = traj_ex.target_pos
    fig, ax = plt.subplots(1)
    sns.set_style("white")
    for traj in pos:
        if len(trajectory_objects_list) < 60:
            alpha=0.4
        else:
            alpha=0.02
#        ax.plot(traj[:, 0], traj[:, 1], lw=2, alpha=1)
        ax.axis([0,1,0.151,-0.151])  # slight y padding for graphs
    title_append = r""" $T_max$ = {0} secs, $\beta = {2}$, $f = {3}$, $wtf = {4}$.
                """.format(traj_ex.Tmax, len(trajectory_objects_list), traj_ex.beta, traj_ex.rf, traj_ex.wtf)
                
    # draw heater
    if traj_ex.target_pos is not None:
        heaterCircle, detectCircle = draw_heaters(traj_ex.target_pos, traj_ex.detect_thresh)
        ax.add_artist(heaterCircle)
        ax.add_artist(detectCircle)
    
    # draw cage
    cage = draw_cage()
    ax.add_patch(cage)

    
    # plot shwag
    title_append = title_append + """<Tfind> = {0:}, Sourcefinds = {1}/(n = {2})""".format(Tfind_avg, sum(target_finds), len(trajectory_objects_list))
    plt.title("Agent trajectories" + title_append, fontsize=14)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.axis(traj_ex.boundary)
    sns.set_style("white")
    
    # save before overlaying heatmap
    plt.savefig("./figs/Trajectories b{beta} f{rf} wf{wtf} bounce {bounce} N{total_trajectories}.png".format(beta=traj_ex.beta, rf=traj_ex.rf, wtf=traj_ex.wtf, bounce=traj_ex.bounce, total_trajectories=len(trajectory_objects_list)))
    
    ## Position heatmap
    if heatmap is True:
        pos_flat = np.array(list(itertools.chain.from_iterable(pos)))
        # crunch the data
        counts, xedges, yedges = np.histogram2d(pos_flat[:,0], pos_flat[:,1], bins=(100,30), range=[[0, 1], [-0.15, .15]])
        
        # counts needs to be transposed to use pcolormesh     
        counts = counts.T
        
        MaxVal = len(trajectory_objects_list)/2
        if len(trajectory_objects_list) > 100:
            plt.cla()
        heatmap = ax.pcolormesh(xedges, yedges, counts, cmap=plt.cm.Oranges, vmin=0, vmax=MaxVal)
#        plt.gca().invert_yaxis()  # hack to match y axis convention --- now unneeded?
        ax.set_ylim(traj_ex.boundary[2:])
        
        # overwrite previous plot schwag
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel('Counts')
        plt.title("Agent Trajectories Heatmap (n = {})".format(len(trajectory_objects_list)))
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.savefig("./figs/Trajectories heatmap beta{beta}_f{rf}_wf{wtf}_bounce {bounce} N{total_trajectories}.png".format(beta=traj_ex.beta, rf=traj_ex.rf, wtf=traj_ex.wtf, bounce=traj_ex.bounce, total_trajectories=len(trajectory_objects_list)))
        plt.show()


def stateHistograms(pos, velos, accels, trajectory_objects_list):
    fig = plt.figure(4, figsize=(9, 8))
    gs1 = gridspec.GridSpec(2, 2)
    axs = [fig.add_subplot(ss) for ss in gs1]
    fig.suptitle("Agent Model Flight Distributions", fontsize=14)
    
    # position distributions
    pos_all = np.concatenate(pos, axis=0)
    pos_binwidth = .01
    
    # X pos
    xpos_min, xpos_max = 0., 1.
    xpos_counts, xpos_bins = np.histogram(pos_all[:, 0], bins=np.linspace(xpos_min, xpos_max, (xpos_max-xpos_min) / pos_binwidth))
    xpos_counts_n = xpos_counts.astype(int) / float(xpos_counts.size)
    axs[0].plot(xpos_bins[:-1]+pos_binwidth/2, xpos_counts_n, lw=2)
    axs[0].bar(xpos_bins[:-1], xpos_counts_n, xpos_bins[1]-xpos_bins[0], facecolor='blue', linewidth=0, alpha=0.1)
    axs[0].set_title("Upwind ($x$) Position Distributions", fontsize=12)
    axs[0].set_xlabel("Position ($m$)")
    axs[0].set_ylabel("Probability")
    
    # Y pos
    ypos_min, ypos_max = -0.15, 0.15
    ypos_counts, ypos_bins = np.histogram(pos_all[:, 1], bins=np.linspace(ypos_min, ypos_max, (ypos_max-ypos_min)/pos_binwidth))
    ypos_counts_n = ypos_counts/ ypos_counts.astype(float).sum()
    axs[1].plot(ypos_bins[:-1]+pos_binwidth/2, ypos_counts_n, lw=2)
    axs[1].set_xlim(ypos_min+pos_binwidth/2, ypos_max-pos_binwidth/2)  # hack to hide gaps
    axs[1].fill_between(ypos_bins[:-1]+pos_binwidth/2, 0, ypos_counts_n, facecolor='blue', alpha=0.1)
    axs[1].set_title("Cross-wind ($y$) Position Distributions")
    axs[1].set_xlabel("Position ($m$)")
    axs[1].set_ylabel("Probability")
    

    ## Velo distributions
    velo_all = np.concatenate(velos, axis=0)
    vmin, vmax = -1.0, 1.
    velo_bindwidth = 0.02
    
    # vx component
    vx_counts, vx_bins = np.histogram(velo_all[:, 0], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    vx_counts_n = vx_counts / vx_counts.astype(float).sum()
    axs[2].plot(vx_bins[:-1], vx_counts_n, label="$\dot{x}$")
    axs[2].fill_between(vx_bins[:-1], 0, vx_counts_n, facecolor='blue', alpha=0.1)
    # vy component
    vy_counts, vy_bins = np.histogram(velo_all[:, 1], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    vy_counts_n= vy_counts / vy_counts.astype(float).sum()
    axs[2].plot(vy_bins[:-1], vy_counts_n, label="$\dot{y}$")
    axs[2].fill_between(vy_bins[:-1], 0, vy_counts_n, facecolor='green', alpha=0.1)
    # |v|
    velo_all_magn = []
    for v in velo_all:
        velo_all_magn.append(np.linalg.norm(v))
    vabs_counts, vabs_bins = np.histogram(velo_all_magn, bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    vabs_counts_n = vabs_counts / vabs_counts.astype(float).sum()
    axs[2].plot(vabs_bins[:-1], vabs_counts_n, label='$|\mathbf{v}|$', color=sns.desaturate("black", .4), lw=2)
    axs[2].fill_between(vabs_bins[:-1], 0, vabs_counts_n, facecolor='yellow', alpha=0.1)
    
    axs[2].set_title("Velocity Distributions")
    axs[2].set_xlabel("Velocity ($m/s$)", fontsize=12)
    axs[2].set_ylabel("Probability", fontsize=12)
    axs[2].legend(fontsize=14)
    
    ## Acceleration distributions
    accel_all = np.concatenate(accels, axis=0)
    
    amin, amax = -10., 10
    accel_binwidth = 0.2
    
    # ax component
    ax_counts, ax_bins = np.histogram(accel_all[:, 0], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    ax_counts_n = ax_counts / ax_counts.astype(float).sum()
    axs[3].plot(ax_bins[:-1], ax_counts_n, label="$\ddot{x}$", lw=2)
    axs[3].fill_between(ax_bins[:-1], 0, ax_counts_n, facecolor='blue', alpha=0.1)
    # ay component
    ay_counts, ay_bins = np.histogram(accel_all[:, 1], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    ay_counts_n = ay_counts / ay_counts.astype(float).sum()
    axs[3].plot(ay_bins[:-1], ay_counts_n, label="$\ddot{y}$", lw=2)
    axs[3].fill_between(ay_bins[:-1], 0, ay_counts_n, facecolor='green', alpha=0.1)
    # |a|
    accel_all_magn = []
    for a in accel_all:
        accel_all_magn.append(np.linalg.norm(a))
    aabs_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    aabs_counts_n = aabs_counts/ aabs_counts.astype(float).sum()
    axs[3].plot(aabs_bins[:-1], aabs_counts_n, label='$|\mathbf{a}|$', color=sns.desaturate("black", .4), lw=2)
    axs[3].fill_between(aabs_bins[:-1], 0, aabs_counts_n, facecolor='yellow', alpha=0.1)
    axs[3].set_title("Acceleration Distribution")
    axs[3].set_xlabel("Acceleration ($m^s/s$)")
    axs[3].set_ylabel("Probability")
    axs[3].legend(fontsize=14)


    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])  # overlapping text hack
    plt.savefig("./figs/Agent Distributions b {beta},f {rf},wf {wtf},bounce {bounce},N {total_trajectories}.png".format(beta=trajectory_objects_list[0].beta, rf=trajectory_objects_list[0].rf, wtf=trajectory_objects_list[0].wtf, bounce=trajectory_objects_list[0].bounce, total_trajectories=len(trajectory_objects_list)))
    
    return xpos_counts_n, ypos_bins, ypos_counts, ypos_counts_n, vx_counts_n


if __name__ == '__main__':
    import trajectory_stats
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajectory_stats.main(total_trajectories=100, beta=4e-6, wallF=(80, 1e-4), plotting = False, wtf=7e-7, rf=4e-6, Tmax=10, bounce=None)
        
    trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list, heatmap=True)
    xpos_counts_n, ypos_bins, ypos_counts, ypos_counts_n, vx_counts_n = stateHistograms(pos, velos, accels, trajectory_objects_list)