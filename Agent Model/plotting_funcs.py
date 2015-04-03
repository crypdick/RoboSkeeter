# -*- coding: utf-8 -*-
"""
Flight trajectory plotting functions

Created on Fri Mar 20 12:27:04 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
from matplotlib import pyplot as plt
import matplotlib as mpl
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
        plt.plot(traj[:, 0], traj[:, 1], lw=2, alpha=0.9)
        ax = plt.gca()
        ax.axis([0,1,-0.151,0.151])  # slight y padding for graphs
    title_append = r"""
    $T_max$ = {0} secs, $\beta = {2}$, $f = {3}$, $wtf = {4}$.
                """.format(traj_ex.Tmax, len(trajectory_objects_list), traj_ex.beta, traj_ex.f0, traj_ex.wf0)
    if target_pos is not None:
        plt.scatter(traj_ex.target_pos[0], traj_ex.target_pos[1], s=150, c='r', marker="*")
        plt.axis(traj_ex.boundary)
        title_append = title_append + """\n
                <Tfind> = {0}, Sourcefinds = {1}/(n = {2})""".format(Tfind_avg, sum(target_finds), len(trajectory_objects_list))
    plt.title("Agent trajectories" + title_append)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    
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
    plt.hist2d(pos_flat[:,0], pos_flat[:,1], bins=(100, 30), normed=True, cmap='gray', cmax=30, range=[traj_ex.boundary[:2], traj_ex.boundary[2:]])
    plt.colorbar()

    plt.title("Agent Trajectories Heatmap", fontsize=20)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.savefig("./figs/agent trajectories heatmap.png")
    plt.show()


def stateHistograms(pos, velos, accels):
    """Plot distributions of position, velocity, and acceleration"""
    NUMBINS = 30
    xmin, xmax = 0, 1.
    ymin, ymax = -0.3, 0.3
    
    # position distributions
    pos_all = np.concatenate(pos, axis=0)
    pos_dist_fig = plt.figure(2)
    xpos_bins = np.linspace(xmin, xmax, NUMBINS)
    ypos_bins = np.linspace(ymin, ymax, NUMBINS)
    sns.distplot(pos_all[:, 0], 
            bins = xpos_bins,
#            normed=True,
            hist=True,
            kde_kws={ "lw": 3, "label": '$x$', "bw":"0.01"},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);
    sns.distplot(pos_all[:, 1],
            bins=ypos_bins,
            hist=True,
            kde_kws={ "lw": 3, "label": '$y$', "bw":"0.01"},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);    
    plt.title("Position distributions", fontsize=20)
    plt.xlabel("Coordinate", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig("./figs/position distributions histo.png")

    ## Velo distributions
    velo_all = np.concatenate(velos, axis=0)
    vmin, vmax = -0.4, 0.4
    velo_bins = np.linspace(vmin, vmax, NUMBINS)
    velo_dist_fig = plt.figure(3)
    # vx component    
    sns.distplot(velo_all[:, 0],
            hist=True,
            kde_kws={ "lw": 3, "label": "$\dot{x}$"},
            hist_kws={"histtype": "stepfilled", "normed":"True"},
            norm_hist=True);
    # ay component
    sns.distplot(velo_all[:, 1],
            hist=True,
            kde_kws={ "lw": 3, "label": "$\dot{y}$"},
            hist_kws={"histtype": "stepfilled", "normed":"True"},
            norm_hist=True);
    velo_all_magn = []
    for v in velo_all:
        velo_all_magn.append(np.linalg.norm(v))
    # |v|
    sns.distplot(velo_all_magn,
            hist=True,
            kde_kws={ "lw": 3, "label": '$|\mathbf{v}|$'},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);
    plt.title("Velocity distributions", fontsize=20)
    plt.xlabel("Velocity", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend(fontsize=14)  
    plt.savefig("./figs/velocity distributions histo.png")
    
    ## Acceleration distributions
    accel_all = np.concatenate(accels, axis=0)
    accel_dist_fig = plt.figure(5)
    # ax component
    sns.distplot(accel_all[:, 0],
            hist=True,
            kde_kws={ "lw": 3, "label": '$\ddot{x}$'},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);
    # ay component
    sns.distplot(accel_all[:, 1],
            hist=True,
            kde_kws={ "lw": 3, "label": '$\ddot{y}$'},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);
    accel_all_magn = []
    for a in accel_all:
        accel_all_magn.append(np.linalg.norm(a))
    # |a|
    sns.distplot(accel_all_magn,
            hist=True,
            kde_kws={ "lw": 3, "label": '$\| \mathbf{a} \|$'},
            hist_kws={"histtype": "stepfilled", "normed": "True"},
            norm_hist=True);
    plt.title("Acceleration Distribution", fontsize=20)
    plt.xlabel("Acceleration", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.legend(fontsize=14)  
    plt.savefig("./figs/acceleration distributions histo.png")



if __name__ == '__main__':
    import trajectory_stats
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajectory_stats.main(total_trajectories=5, target_pos="left")
    