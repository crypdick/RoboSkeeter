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


def trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list):
    """"Plot all the trajectories into a single arena"""
    traj_ex = trajectory_objects_list[0]
    target_pos = traj_ex.target_pos
    agent_paths_fig = plt.figure(1)
    for traj in pos:
        plt.plot(traj[:, 0], traj[:, 1], lw=2, alpha=0.4)
        ax = plt.gca()
        ax.axis([0,1,0.151,-0.151])  # slight y padding for graphs
    title_append = r"""
    $T_max$ = {0} secs, $\beta = {2}$, $f = {3}$, $wtf = {4}$.
                """.format(traj_ex.Tmax, len(trajectory_objects_list), traj_ex.beta, traj_ex.f0, traj_ex.wf0)
    # draw heater
    if target_pos is not None:
#        plt.scatter(traj_ex.target_pos[0], traj_ex.target_pos[1], s=150, c='r', marker="o")
        heaterCircle = plt.Circle((traj_ex.target_pos[0], traj_ex.target_pos[1],), 0.003175, color='r')  # 0.003175 is diam of our heater
        detectCircle = plt.Circle((traj_ex.target_pos[0], traj_ex.target_pos[1],), traj_ex.detect_thresh, color='gray', fill=False)
        ax.add_artist(heaterCircle)
        ax.add_artist(detectCircle)
        title_append = title_append + """\n
                <Tfind> = {0}, Sourcefinds = {1}/(n = {2})""".format(Tfind_avg, sum(target_finds), len(trajectory_objects_list))
    plt.title("Agent trajectories" + title_append)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.axis(traj_ex.boundary)
    sns.set_style("white")
    
    # draw cage
    cage_midX, cage_midY = 0.1524, 0.
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((cage_midX - 0.0381, cage_midY - 0.0381), 0.0762, 0.0762, facecolor='none'))
    # draw walls
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((0, -0.15), 1., 0.3, facecolor='none', lw=1))

    plt.savefig("./figs/agent trajectories.png")
    plt.show()
    
    ## Position heatmap
    pos_flat = np.array(list(itertools.chain.from_iterable(pos)))
    plt.hist2d(pos_flat[:,0], pos_flat[:,1], bins=(100, 30), normed=True, cmap='gray', cmax=30, range=[traj_ex.boundary[:2], [traj_ex.boundary[3], traj_ex.boundary[2]]])
    plt.gca().invert_yaxis()  # fix for y axis convention
    plt.colorbar()

    plt.title("Agent Trajectories Heatmap")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("./figs/agent trajectories heatmap.png")
    plt.show()
    

def stateHistograms(pos, velos, accels):
    fig = plt.figure(2, figsize=(8, 15))
    gs1 = gridspec.GridSpec(4, 1)
    axs = [fig.add_subplot(ss) for ss in gs1]
    fig.suptitle("Agent Model Flight Distributions", fontsize=14)
    
    # position distributions
    pos_all = np.concatenate(pos, axis=0)
    pos_binwidth = .001
    
    # X pos
    xpos_min, xpos_max = 0., 1.
    counts, bins = np.histogram(pos_all[:, 0], bins=np.linspace(xpos_min, xpos_max, (xpos_max-xpos_min) / pos_binwidth))
    counts_n= counts / float(len(counts))
    axs[0].plot(bins[:-1], counts_n, lw=2)
    axs[0].set_title("Upwind ($x$) Position Distributions")
    axs[0].set_xlabel("Position ($m$)")
    axs[0].set_ylabel("Probability")
    axs[0].legend()
    
    # Y pos
    ypos_min, ypos_max = -0.15, 0.15
    counts, bins = np.histogram(pos_all[:, 1], bins=np.linspace(ypos_min, ypos_max, (ypos_max-ypos_min)/pos_binwidth))
    counts_n= counts / float(len(counts))
    axs[1].plot(bins[:-1], counts_n, lw=2)
    axs[1].set_title("Cross-wind ($y$) Position Distributions")
    axs[1].set_xlabel("Position ($m$)")
    axs[1].set_ylabel("Probability")
    axs[1].legend()
    

    ## Velo distributions
    velo_all = np.concatenate(velos, axis=0)
    vmin, vmax = -0.4, 0.4
    velo_bindwidth = 0.01
    
    # vx component
    counts, bins = np.histogram(velo_all[:, 0], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    counts_n= counts/float(len(counts))
    axs[2].plot(bins[:-1], counts_n, label="$\dot{x}$")
    # vy component
    counts, bins = np.histogram(velo_all[:, 1], bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    counts_n= counts/float(len(counts))
    axs[2].plot(bins[:-1], counts_n, label="$\dot{y}$")
    # |v|
    velo_all_magn = []
    for v in velo_all:
        velo_all_magn.append(np.linalg.norm(v))
    counts, bins = np.histogram(velo_all_magn, bins=np.linspace(vmin, vmax, (vmax-vmin)/velo_bindwidth))
    counts_n= counts/float(len(counts))
    axs[2].plot(bins[:-1], counts_n, label='$|\mathbf{v}|$', color=sns.desaturate("black", .4), lw=2)
    
    axs[2].set_title("Velocity Distributions")
    axs[2].set_xlabel("Velocity ($m/s$)")
    axs[2].set_ylabel("Probability")
    axs[2].legend()
    
    ## Acceleration distributions
    accel_all = np.concatenate(accels, axis=0)
    
    amin, amax = -10., 10
    accel_binwidth = 0.1
    
    # ax component
    counts, bins = np.histogram(accel_all[:, 0], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    counts_n= counts/float(len(counts))
    axs[3].plot(bins[:-1], counts_n, label="$\ddot{x}$", lw=2)
    # ay component
    counts, bins = np.histogram(accel_all[:, 1], bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    counts_n= counts/float(len(counts))
    axs[3].plot(bins[:-1], counts_n, label="$\ddot{y}$", lw=2)
    # |a|
    accel_all_magn = []
    for a in accel_all:
        accel_all_magn.append(np.linalg.norm(a))
    counts, bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, (amax-amin)/accel_binwidth))
    counts_n= counts/float(len(counts))
    axs[3].plot(bins[:-1], counts_n, label='$|\mathbf{a}|$', color=sns.desaturate("black", .4), lw=2)
    axs[3].set_title("Acceleration Distribution")
    axs[3].set_xlabel("Acceleration ($m^s/s$)")
    axs[3].set_ylabel("Probability")
    axs[3].legend()


    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])  # overlapping text hack
    plt.savefig("./figs/Agent Distributions.png")


if __name__ == '__main__':
    import trajectory_stats
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajectory_stats.main(total_trajectories=100, target_pos="left")
    