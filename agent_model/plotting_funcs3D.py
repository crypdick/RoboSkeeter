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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_palette("muted", 8)


def plot_single_trajectory(dynamics, metadata, plot_kwargs=None):
    # plot an individual trajectory
    # TODO: make 3D
    try:
        plt.plot(dynamics['position_x'], dynamics['position_y'], lw=2, alpha=0.5)
    except KeyError:
        print dynamics.keys()
        pass
    currentAxis = plt.gca()
    cage = draw_cage()
    currentAxis.add_patch(cage)
    currentAxis.axis(metadata['boundary'][0:4])
    if metadata['heater_position'] is not None:
        heaterCircle, detectCircle = draw_heaters(metadata['heater_position'], metadata['detection_threshold'])
        currentAxis.add_artist(heaterCircle)
        currentAxis.add_artist(detectCircle)
    plt.gca().set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(15, 4.5)

    plt.title(plot_kwargs['title'] + plot_kwargs['titleappend'], fontsize=20)
    plt.xlabel("Upwind/$x$ (meters)", fontsize=14)
    plt.ylabel("Crosswind/$y$ (meters)", fontsize=14)
    plt.savefig(
        "./figs/indiv_traj beta{beta}_f{rf}_wf{wtf}.svg".format(beta=metadata['beta'], rf=metadata['randomF_strength'],
                                                                wtf=metadata['wtF']), format="svg")


def draw_cage():
    # makes a little box where the cage is
    cage_midX, cage_midY = 0.1524, 0.  # TODO: turn to absolute boundaries and put in metadata
    return plt.Rectangle((cage_midX - 0.0381, cage_midY - 0.0381), 0.0762, 0.0762,
                         facecolor='none')  # FIXME get rid of hardcoded number


def draw_heaters(heater_position, detect_thresh):
    """ draws a circle where the heater is
    heater_position vector is [x,y, zmin, zmax, diam]
    """
    heaterCircle = plt.Circle((heater_position[0], heater_position[1],), heater_position[4] / 2, color='r')
    detectCircle = plt.Circle((heater_position[0], heater_position[1],), detect_thresh, color='gray', fill=False,
                              linestyle='dashed')

    return heaterCircle, detectCircle


# def trajectory_plots(ensemble, metadata, plot_kwargs=None):
#     """"Plot all the trajectories into a single arena"""
#     fig, ax = plt.subplots(1)
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
##                """.format(metadata['time_max'], metadata['total_trajectories'], metadata['beta'], metadata['randomF_strength'], metadata['wtF'])
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
#    plt.savefig("./figs/Trajectories b{beta} f{rf} wf{wtf} N{total_trajectories}.png"\
#        .format(beta=metadata['beta'], rf=metadata['randomF_strength'], wtf=metadata['wtF'], total_trajectories=metadata['total_trajectories']))
#    
############################################## NEW HEATMAP #######################
#    ## Position heatmap
#    if plot_kwargs['heatmap']:
#        with sns.axes_style("white"):
#            hextraj = sns.jointplot('position_x', 'position_y', ensemble, size=10)#, ylim=(-.15, .15))
#            hextraj.plot_marginals(sns.distplot, kde=False)
#            hextraj.plot_joint(plt.hexbin, vmax=30, extent = [metadata['boundary'][0], metadata['boundary'][1], metadata['boundary'][3], metadata['boundary'][2]])#joint_kws={'gridsize':(100,30), })
#            hextraj.ax_joint.set_aspect('equal')
#            hextraj.ax_joint.invert_yaxis()  # hack to match y axis convention 
#            cax = hextraj.fig.add_axes([1, .25, .04, .5])
#            plt.colorbar(cax=cax)
#            
#            if metadata['heater_position'] is not None:
#                heaterCircle, detectCircle = draw_heaters(metadata['heater_position'], metadata['detection_threshold'])
#                hextraj.ax_joint.add_artist(heaterCircle)
#                hextraj.ax_joint.add_artist(detectCircle)
#        #    
#            # draw cage
#            cage = draw_cage()
#            hextraj.ax_joint.add_patch(cage)
#        plt.savefig("./figs/Trajectories sns heatmap beta{beta}_f{rf}_wf{wtf}_N{total_trajectories}.png".format(beta=metadata['beta'], rf=metadata['randomF_strength'], wtf=metadata['wtF'], total_trajectories=metadata['total_trajectories']))
#            
########## OLD HEATMAP ##############################################
# crunch the data
#        counts, xedges, yedges = np.histogram2d(ensemble['position_x'], ensemble['position_y'], bins=(100,30), range=[[0, 1], [-0.15, .15]])
# only considering trajectories between 0.25 - 0.95 m in x direction

def heatmaps(ensemble, metadata):
    fig, ax = plt.subplots(1)
    ensemble = ensemble.loc[(ensemble['position_x'] > 0.25) & (ensemble['position_x'] < 0.95)]
    counts, xedges, yedges = np.histogram2d(ensemble['position_x'], ensemble['position_y'], bins=(100, 30),
                                            range=[[0.25, 0.95], [-0.127, .127]])



    # counts needs to be transposed to use pcolormesh
    counts = counts.T
    probs = counts / ensemble.size

    #        MaxVal = metadata['total_trajectories']/2
    #        if metadata['total_trajectories'] > 100:
    #            plt.cla()
    heatmap = ax.pcolormesh(xedges, yedges, probs, cmap=plt.cm.Oranges, vmin=0.)  # , vmax=.2)
    if metadata['heater_position'] is not None:
        heaterCircle, detectCircle = draw_heaters(metadata['heater_position'], metadata['detection_threshold'])
        ax.add_artist(heaterCircle)
        ax.add_artist(detectCircle)
    #        #
    #            # draw cage
    #            cage = draw_cage()
    #            hextraj.ax_joint.add_patch(cage)
    ax.set_aspect('equal')
    #        ax.invert_yaxis()  # hack to match y axis convention --- now unneeded?
    ax.set_ylim([0.127, -.127])  # keep positive first to invert axis
    ax.set_xlim([0.25, 0.95])

    # overwrite previous plot schwag
    cbar = plt.colorbar(heatmap, shrink=0.5, pad=0.05)
    cbar.ax.set_ylabel('Probability')
    plt.title("Agent trajectories 2D position histogram (n = {})".format(metadata['total_trajectories']))
    plt.xlabel("Upwind/$x$ (meters)")
    plt.ylabel("Crosswind/$y$ (meters)")
    plt.savefig(
        "./figs/Trajectories heatmap beta{beta}_f{rf}_wf{wtf}_N{total_trajectories}.svg".format(beta=metadata['beta'],
                                                                                                rf=metadata['randomF_strength'],
                                                                                                wtf=metadata['wtF'],
                                                                                                total_trajectories=
                                                                                                metadata[
                                                                                                    'total_trajectories']),
        format="svg")
    plt.show()


def stateHistograms(
        ensemble,
        metadata,
        plot_kwargs=None,
        titleappend='',
        upw_ensemble="none",
        downw_ensemble="none"):
    statefig = plt.figure()  # , figsize=(9, 8))
    gs1 = gridspec.GridSpec(2, 3)
    axs = [statefig.add_subplot(ss) for ss in gs1]
    statefig.suptitle("Agent Model Flight Distributions" + titleappend, fontsize=14)
    # position distributions
    #    pos_all = np.concatenate(pos, axis=0)
    pos_binwidth = .01

    # X pos
    xpos_min, xpos_max = 0.25, .85
    xpos_counts, xpos_bins = np.histogram(ensemble['position_x'],
                                          bins=np.linspace(xpos_min, xpos_max, (xpos_max - xpos_min) / pos_binwidth))
    xpos_counts = xpos_counts.astype(float)
    xpos_counts_norm = xpos_counts / xpos_counts.sum()
    axs[0].plot(xpos_bins[:-1] + pos_binwidth / 2, xpos_counts_norm, lw=2, label="Full")
    axs[0].bar(xpos_bins[:-1], xpos_counts_norm, xpos_bins[1] - xpos_bins[0], facecolor='blue', linewidth=0, alpha=0.1)
    if type(downw_ensemble) != str:
        xpos_counts, xpos_bins = np.histogram(downw_ensemble['position_x'], bins=np.linspace(xpos_min, xpos_max, (
        xpos_max - xpos_min) / pos_binwidth))
        xpos_counts = xpos_counts.astype(float)
        xpos_counts_norm = xpos_counts / xpos_counts.sum()
        axs[0].plot(xpos_bins[:-1] + pos_binwidth / 2, xpos_counts_norm, lw=2, label="Downwind")
        axs[0].bar(xpos_bins[:-1], xpos_counts_norm, xpos_bins[1] - xpos_bins[0], facecolor='blue', linewidth=0,
                   alpha=0.1)
    if type(upw_ensemble) != str:
        xpos_counts, xpos_bins = np.histogram(upw_ensemble['position_x'], bins=np.linspace(xpos_min, xpos_max, (
        xpos_max - xpos_min) / pos_binwidth))
        xpos_counts = xpos_counts.astype(float)
        xpos_counts_norm = xpos_counts / xpos_counts.sum()
        axs[0].plot(xpos_bins[:-1] + pos_binwidth / 2, xpos_counts_norm, lw=2, label="Upwind")
        axs[0].bar(xpos_bins[:-1], xpos_counts_norm, xpos_bins[1] - xpos_bins[0], facecolor='blue', linewidth=0,
                   alpha=0.1)
        axs[0].legend()
    axs[0].set_title("Upwind ($x$) Position Distributions", fontsize=12)
    axs[0].set_xlim(xpos_min + pos_binwidth / 2, xpos_max - pos_binwidth / 2)
    axs[0].set_xlabel("Position ($m$)")
    axs[0].set_ylabel("Probability")

    # Y pos
    ypos_min, ypos_max = -0.127, 0.127
    ypos_counts, ypos_bins = np.histogram(ensemble['position_y'],
                                          bins=np.linspace(ypos_min, ypos_max, (ypos_max - ypos_min) / pos_binwidth))
    ypos_counts = ypos_counts.astype(float)
    ypos_counts_norm = ypos_counts / ypos_counts.sum()
    axs[1].plot(ypos_bins[:-1] + pos_binwidth / 2, ypos_counts_norm, lw=2, label="Full")
    axs[1].set_xlim(ypos_min + pos_binwidth / 2, ypos_max - pos_binwidth / 2)  # hack to hide gaps
    axs[1].fill_between(ypos_bins[:-1] + pos_binwidth / 2, 0, ypos_counts_norm, facecolor='blue', alpha=0.1)
    if type(downw_ensemble) != str:  # TODO: is this alright with 3D flights?
        ypos_counts, ypos_bins = np.histogram(downw_ensemble['position_y'], bins=np.linspace(ypos_min, ypos_max, (
        ypos_max - ypos_min) / pos_binwidth))
        ypos_counts = ypos_counts.astype(float)
        ypos_counts_norm = ypos_counts / ypos_counts.sum()
        axs[1].plot(ypos_bins[:-1] + pos_binwidth / 2, ypos_counts_norm, lw=2, label="Full")
        axs[1].set_xlim(ypos_min + pos_binwidth / 2, ypos_max - pos_binwidth / 2)  # hack to hide gaps
        axs[1].fill_between(ypos_bins[:-1] + pos_binwidth / 2, 0, ypos_counts_norm, facecolor='blue', alpha=0.1)
    if type(upw_ensemble) != str:
        ypos_counts, ypos_bins = np.histogram(upw_ensemble['position_y'], bins=np.linspace(ypos_min, ypos_max, (
        ypos_max - ypos_min) / pos_binwidth))
        ypos_counts = ypos_counts.astype(float)
        ypos_counts_norm = ypos_counts / ypos_counts.sum()
        axs[1].plot(ypos_bins[:-1] + pos_binwidth / 2, ypos_counts_norm, lw=2, label="Full")
        axs[1].set_xlim(ypos_min + pos_binwidth / 2, ypos_max - pos_binwidth / 2)  # hack to hide gaps
        axs[1].fill_between(ypos_bins[:-1] + pos_binwidth / 2, 0, ypos_counts_norm, facecolor='blue', alpha=0.1)
        axs[1].legend()
    axs[1].set_title("Cross-wind ($y$) Position Distributions")
    axs[1].set_xlabel("Position ($m$)")
    axs[1].set_ylabel("Probability")

    #    Z
    zpos_min, zpos_max = 0, 0.254
    zpos_counts, zpos_bins = np.histogram(ensemble['position_z'],
                                          bins=np.linspace(zpos_min, zpos_max, (zpos_max - zpos_min) / pos_binwidth))
    zpos_counts = zpos_counts.astype(float)
    zpos_counts_norm = zpos_counts / zpos_counts.sum()
    axs[2].plot(zpos_bins[:-1] + pos_binwidth / 2, zpos_counts_norm, lw=2, label="Full")
    axs[2].set_xlim(zpos_min + pos_binwidth / 2, zpos_max - pos_binwidth / 2)  # hack to hide gaps
    axs[2].fill_between(zpos_bins[:-1] + pos_binwidth / 2, 0, zpos_counts_norm, facecolor='blue', alpha=0.1)
    if type(downw_ensemble) != str:
        zpos_counts, zpos_bins = np.histogram(downw_ensemble['position_y'], bins=np.linspace(zpos_min, zpos_max, (
        zpos_max - zpos_min) / pos_binwidth))
        zpos_counts = zpos_counts.astype(float)
        zpos_counts_norm = zpos_counts / zpos_counts.sum()
        axs[2].plot(zpos_bins[:-1] + pos_binwidth / 2, zpos_counts_norm, lw=2, label="Full")
        axs[2].set_xlim(zpos_min + pos_binwidth / 2, zpos_max - pos_binwidth / 2)  # hack to hide gaps
        axs[2].fill_between(zpos_bins[:-1] + pos_binwidth / 2, 0, zpos_counts_norm, facecolor='blue', alpha=0.1)
    if type(upw_ensemble) != str:
        zpos_counts, zpos_bins = np.histogram(upw_ensemble['position_y'], bins=np.linspace(zpos_min, zpos_max, (
        zpos_max - zpos_min) / pos_binwidth))
        zpos_counts = zpos_counts.astype(float)
        zpos_counts_norm = zpos_counts / zpos_counts.sum()
        axs[2].plot(zpos_bins[:-1] + pos_binwidth / 2, zpos_counts_norm, lw=2, label="Full")
        axs[2].set_xlim(zpos_min + pos_binwidth / 2, zpos_max - pos_binwidth / 2)  # hack to hide gaps
        axs[2].fill_between(zpos_bins[:-1] + pos_binwidth / 2, 0, zpos_counts_norm, facecolor='blue', alpha=0.1)
        axs[2].legend()
    axs[2].set_title("Elevation ($z$) Position Distributions")
    axs[2].set_xlabel("Position ($m$)")
    axs[2].set_ylabel("Probability")

    ## Velo distributions
    vmin, vmax = -.7, .7
    velo_bindwidth = 0.02

    # vx component
    vx_counts, vx_bins = np.histogram(ensemble['velocity_x'],
                                      bins=np.linspace(vmin, vmax, (vmax - vmin) / velo_bindwidth))
    vx_counts = vx_counts.astype(float)
    vx_counts_n = vx_counts / vx_counts.sum()
    axs[3].plot(vx_bins[:-1], vx_counts_n, label="$\dot{x}$")
    axs[3].fill_between(vx_bins[:-1], 0, vx_counts_n, facecolor='blue', alpha=0.1)
    # vy component
    vy_counts, vy_bins = np.histogram(ensemble['velocity_y'],
                                      bins=np.linspace(vmin, vmax, (vmax - vmin) / velo_bindwidth))
    vy_counts = vy_counts.astype(float)
    vy_counts_n = vy_counts / vy_counts.sum()
    axs[3].plot(vy_bins[:-1], vy_counts_n, label="$\dot{y}$")
    axs[3].fill_between(vy_bins[:-1], 0, vy_counts_n, facecolor='green', alpha=0.1)
    # vz component
    vz_counts, vz_bins = np.histogram(ensemble['velocity_z'],
                                      bins=np.linspace(vmin, vmax, (vmax - vmin) / velo_bindwidth))
    vz_counts = vz_counts.astype(float)
    vz_counts_n = vz_counts / vz_counts.sum()
    axs[3].plot(vz_bins[:-1], vz_counts_n, label="$\dot{z}$")
    axs[3].fill_between(vz_bins[:-1], 0, vz_counts_n, facecolor='red', alpha=0.1)
    # |v|
    velo_stack = np.vstack((ensemble.velocity_x.values, ensemble.velocity_y.values, ensemble.velocity_z.values))
    velo_all_magn = np.linalg.norm(velo_stack, axis=0)
    # abs_velo_vectors = np.array([ensemble.velocity_x.values, ensemble.velocity_y.values, ensemble.velocity_z.values]).T
    # velo_all_magn = []
    # for vector in abs_velo_vectors:
    #     velo_all_magn.append(np.linalg.norm(vector))
    vabs_counts, vabs_bins = np.histogram(velo_all_magn, bins=np.linspace(vmin, vmax, (vmax - vmin) / velo_bindwidth))
    vabs_counts = vabs_counts.astype(float)
    vabs_counts_n = vabs_counts / vabs_counts.sum()
    axs[3].plot(vabs_bins[:-1], vabs_counts_n, label='$|\mathbf{v}|$', color=sns.desaturate("black", .4), lw=2)
    axs[3].fill_between(vabs_bins[:-1], 0, vabs_counts_n, facecolor='yellow', alpha=0.1)

    axs[3].set_title("Velocity Distributions")
    axs[3].set_xlabel("Velocity ($m/s$)", fontsize=12)
    axs[3].set_ylabel("Probability", fontsize=12)
    axs[3].legend(fontsize=14)

    ## Acceleration distributions
    #    accel_all = np.concatenate(accels, axis=0)

    amin, amax = -4., 4.
    accel_binwidth = 0.101

    # ax component
    ax_counts, ax_bins = np.histogram(ensemble['acceleration_x'],
                                      bins=np.linspace(amin, amax, (amax - amin) / accel_binwidth))
    ax_counts = ax_counts.astype(float)
    ax_counts_n = ax_counts / ax_counts.sum()
    axs[4].plot(ax_bins[:-1], ax_counts_n, label="$\ddot{x}$", lw=2)
    axs[4].fill_between(ax_bins[:-1], 0, ax_counts_n, facecolor='blue', alpha=0.1)
    # ay component
    ay_counts, ay_bins = np.histogram(ensemble['acceleration_y'],
                                      bins=np.linspace(amin, amax, (amax - amin) / accel_binwidth))
    ay_counts = ay_counts.astype(float)
    ay_counts_n = ay_counts / ay_counts.sum()
    axs[4].plot(ay_bins[:-1], ay_counts_n, label="$\ddot{y}$", lw=2)
    axs[4].fill_between(ay_bins[:-1], 0, ay_counts_n, facecolor='green', alpha=0.1)
    # az component
    az_counts, az_bins = np.histogram(ensemble['acceleration_z'],
                                      bins=np.linspace(amin, amax, (amax - amin) / accel_binwidth))
    az_counts = az_counts.astype(float)
    az_counts_n = az_counts / az_counts.sum()
    axs[4].plot(az_bins[:-1], az_counts_n, label="$\ddot{z}$", lw=2)
    axs[4].fill_between(az_bins[:-1], 0, az_counts_n, facecolor='red', alpha=0.1)
    # |a|
    accel_stack = np.vstack((ensemble.acceleration_x.values, ensemble.acceleration_y.values, ensemble.acceleration_z.values))
    accel_all_magn = np.linalg.norm(accel_stack, axis=0)
    # abs_accel_vectors = np.array(
    #     [ensemble.acceleration_x.values, ensemble.acceleration_y.values, ensemble.acceleration_z.values]).T
    # accel_all_magn = []
    # for vector in abs_accel_vectors:
    #     accel_all_magn.append(np.linalg.norm(vector))
    aabs_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, (amax - amin) / accel_binwidth))
    aabs_counts = aabs_counts.astype(float)
    aabs_counts_n = aabs_counts / aabs_counts.sum()
    axs[4].plot(aabs_bins[:-1], aabs_counts_n, label='$|\mathbf{a}|$', color=sns.desaturate("black", .4), lw=2)
    axs[4].fill_between(aabs_bins[:-1], 0, aabs_counts_n, facecolor='yellow', alpha=0.1)
    axs[4].set_title("Acceleration Distribution")
    axs[4].set_xlabel("Acceleration ($m^s/s$)")
    axs[4].set_ylabel("Probability")
    axs[4].legend(fontsize=14)

    gs1.tight_layout(statefig, rect=[0, 0.03, 1, 0.95])  # overlapping text hack
    plt.savefig(
        "./figs/Agent Distributions b {beta},f {rf},wf {wtf},,N {total_trajectories}.svg".format(
            beta=metadata['beta'],
            rf=metadata['randomF_strength'],
            wtf=metadata['wtF'],
            total_trajectories=metadata['total_trajectories']
        ),
        format='svg')
    plt.show()


#    return xpos_counts_norm, ypos_bins, ypos_counts, ypos_counts_norm, vx_counts_n


def force_violin(ensemble, metadata):
    ensembleF = ensemble.loc[
        (ensemble['position_x'] > 0.25) & (ensemble['position_x'] < 0.95),
            ['totalF_x', 'totalF_y', 'totalF_z',
            'biasF_x', 'biasF_y', 'biasF_z',
            'upwindF_x',
            'wallRepulsiveF_x', 'wallRepulsiveF_y', 'wallRepulsiveF_z',
            'stimF_x', 'stimF_y']] #, 'stimF_z']]== Nans
    # plot Forces
    #    f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    ##    forcefig = plt.figure(5, figsize=(9, 8))
    ##    gs2 = gridspec.GridSpec(2, 2)
    ##    Faxs = [fig.add_subplot(ss) for ss in gs2]
    forcefig = plt.figure()
    #    Faxs1 = forcefig.add_subplot(211)
    #    Faxs2 = forcefig.add_subplot(212)
    sns.violinplot(ensembleF, lw=2, alpha=0.7, palette="Set2")
    #    tF = sns.jointplot('totalF_x', 'totalF_y', ensemble, kind="hex", size=10)
    plt.suptitle("Force distributions")
    #    plt.xticks(range(4,((len(alignments.keys())+1)*4),4), [i[1] for i in medians_sgc], rotation=90, fontsize = 4)
    plt.tick_params(axis='x', pad=4)
    plt.xticks(rotation=40)
    #    remove_border()
    plt.tight_layout(pad=1.8)

    plt.ylabel("Force magnitude distribution (newtons)")
    plt.savefig("./figs/Force Distributions b {beta},f {rf},wf {wtf},N {total_trajectories}.svg".format( \
        beta=metadata['beta'], rf=metadata['randomF_strength'], wtf=metadata['wtF'],
        total_trajectories=metadata['total_trajectories']), format='svg')


def velo_compass_histogram(ensemble, metadata, kind):
    N = 25
    roundbins = np.linspace(0.0, 2 * np.pi, N)

    ensemble['magnitude'] = [np.linalg.norm(x) for x in ensemble.values]
    ensemble['angle'] = np.tan(ensemble['velocity_y'] / ensemble['velocity_x']) % (2 * np.pi)

    if kind == 'global_normalize':
        """what fraction of the total magnitude in all bins were in this bin?
        """
        ensemble['fraction'] = ensemble['magnitude'] / ensemble['magnitude'].sum()
        width = (2 * np.pi) / N

        values, bin_edges = np.histogram(ensemble['angle'], weights=ensemble['fraction'], bins=roundbins)

        compassfig = plt.figure()
        ax = plt.subplot(111, polar=True)

        ax.bar(bin_edges[:-1], values, width=width, linewidth=0)
        plt.xlim(min(bin_edges), max(bin_edges))

        # switch to radian labels
        x_tix = plt.xticks()[0]
        x_labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', \
                    r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
        plt.xticks(x_tix, x_labels, size=20)
        plt.title(
            "Agent velocities 0.25 < x < 0.5, center repulsion on (n = {})".format(metadata['total_trajectories']),
            y=1.1)
        plt.savefig(
            "./figs/Velocity compass, center repulsion on_ beta{beta}_rf{biasF_scale}_wf{wtf}_N{total_trajectories}.svg".format( \
                beta=metadata['beta'], biasF_scale=metadata['randomF_strength'], wtf=metadata['wtF'],
                total_trajectories=metadata['total_trajectories']), format="svg")

    if kind == 'bin_average':
        """for each bin, we want the average magnitude
        select bin range with pandas,
        sum(magnitude) / n_vectors
        """
        bin_mag_avgs = np.zeros(N - 1)
        for i in range(N - 1):
            bin_magnitudes = ensemble.loc[
                ((ensemble['angle'] > roundbins[i]) & (ensemble['angle'] < roundbins[i + 1])), ['magnitude']]
            bin_mag_avg = bin_magnitudes.sum() / bin_magnitudes.size
            bin_mag_avgs[i] = bin_mag_avg


def compass_histogram(vector_name, ensemble, metadata, kind='avg_mag_per_bin', title='', fname=''):
    """TODO: make compass plot total mag per bin, overlay compass arrows for avg magnitude

    :param vector_name:
    :param ensemble:
    :param metadata:
    :param kind:
    :param title:
    :param fname:
    :return:
    """
    N = 25
    roundbins = np.linspace(0.0, 2 * np.pi, N)
    width = (2 * np.pi) / N


    #    if kind == 'global_normalize':
    #        ensemble['fraction'] = ensemble['magnitude'] / ensemble['magnitude'].sum()
    #
    #        values, bin_edges = np.histogram(ensemble['angle'], weights = ensemble['fraction'], bins = roundbins)
    #
    #        compassfig = plt.figure()
    #        ax = plt.subplot(111, polar=True)
    #
    #        ax.bar(bin_edges[:-1], values, width = width, linewidth = 0)
    #        plt.xlim(min(bin_edges), max(bin_edges))
    #
    #        # switch to radian labels
    #        xT = plt.xticks()[0]
    #        xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    #            r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    #        plt.xticks(xT, xL, size = 20)
    #        plt.title("Agent velocities 0.25 < x < 0.5, center repulsion on (n = {})".format(metadata['total_trajectories']), y=1.1)
    #        plt.savefig("./figs/Compass plot , center repulsion on_ beta{beta}_rf{biasF_scale}_wf{wtf}_N{total_trajectories}.svg".format(\
    #            beta=metadata['beta'], biasF_scale=metadata['randomF_strength'], wtf=metadata['wtF'], total_trajectories=metadata['total_trajectories']), format="svg")

    if kind == 'avg_mag_per_bin':
        """for each bin, we want the average magnitude
        select bin range with pandas,
        """
        bin_mag_avgs = np.zeros(N - 1)
        for i in range(N - 1):  # for bin
            # select all magnitudes in the bin
            bin_magnitudes = ensemble.loc[
                ((ensemble['angle'] >= roundbins[i]) & (ensemble['angle'] < roundbins[i + 1])), ['magnitude']]
            # find their mean
            bin_mag_avg = bin_magnitudes.sum() / bin_magnitudes.size
            bin_mag_avgs[i] = bin_mag_avg

        compassfig = plt.figure()
        ax = plt.subplot(111, polar=True)

        ax.bar(roundbins[:-1], bin_mag_avgs, width=width, linewidth=0)
        #        plt.xlim(min(bin_edges), max(bin_edges))

        # switch to radian labels
        x_tix = plt.xticks()[0]
        x_labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', \
              r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
        plt.xticks(x_tix, x_labels, size=20)
        plt.title(title)
        plt.savefig("./figs/Compass {fname}.svg".format(fname=fname), format="svg")


def cylinder(center_x, center_y, z_min, z_max, r=0.01905, n=5):
    '''
    Returns the unit cylinder that corresponds to the curve r.
    INPUTS:  r - a vector of radii
             n - number of coordinates to return for each element in r
             TODO: update with new params

    OUTPUTS: x,y,z - coordinates of points
    
    modified from http://python4econ.blogspot.com/2013/03/matlabs-cylinder-command-in-python.html
    '''
    # TODO: FIXME cylinder not plotting in correct place
    # ensure that r is a column vector
    r = np.atleast_2d(r)
    r_rows, r_cols = r.shape

    if r_cols > r_rows:
        r = r.T

    # find points along x and y axes
    points = np.linspace(0., 2. * np.pi, n + 1)  # generate evenly spaced rads
    x = np.cos(points) * r + center_x
    y = np.sin(points) * r + center_y

    # find points along z axis
    rpoints = np.atleast_2d(np.linspace(z_min, z_max, len(r)))
    z = np.ones((1, n + 1)) * rpoints.T

    return x, y, z


def plot3D_trajectory(ensemble, metadata, plot_kwargs=None):
    '''plotting without coloring
    '''

    fig3D = plt.figure()
    threedee = fig3D.gca(projection='3d')
    threedee.auto_scale_xyz([0., 0.3], [0., 1.], [0., 0.254])
    threedee.plot(ensemble.position_y, ensemble.position_x, ensemble.position_z)

    # # Cylinder
    #
    # # get points from cylinder and plot
    # cx, cy, zmin, zmax, diam, label = cylinder(*metadata['heater_position'])
    # threedee.plot_wireframe(cx, cy, zmax)

    # Note! I set the y axis to be the X data and vice versa
    threedee.set_ylim(0., 1.)
    threedee.set_xlim(0.127, -0.127)
    threedee.invert_xaxis()  # fix for y convention
    threedee.set_zlim(0., 0.254)
    threedee.set_xlabel("Crosswind/$y$ (meters)", fontsize=14)  # remember! x,y switched in plot() above!
    threedee.set_ylabel("Upwind/$x$ (meters)", fontsize=14)
    threedee.set_zlabel("Elevation/$z$ (meters)", fontsize=14)
    threedee.set_title(plot_kwargs['title'] + plot_kwargs['titleappend'], fontsize=20)
    #    plt.savefig("./correlation_figs/{data_name}/{data_name} Trajectory.svg".format(data_name = csv_name), format="svg")
    plt.show()

    # todo: plot cylinder, detect circle, cage


    # if __name__ == '__main__':
    #    import trajectory_stats
    #
    #        # wallF params
    #    wallF_max=5e-7
    #    decay_const = 250
    #
    #    # center repulsion params
    #    b = 4e-1  # determines shape
    #    shrink = 5e-7  # determines size/magnitude
    #
    #    wallF = (b, shrink, wallF_max, decay_const)
    #
    #    ensemble, metadata = trajectory_stats.main()
    #
    #    trajectory_plots(ensemble, metadata, heatmap=True, trajectoryPlot = True)
    #    xpos_counts_norm, ypos_bins, ypos_counts, ypos_counts_norm, vx_counts_n = stateHistograms(ensemble, metadata)
