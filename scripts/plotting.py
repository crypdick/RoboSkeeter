# -*- coding: utf-8 -*-
"""
Flight trajectory plotting functions

Created on Fri Mar 20 12:27:04 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import os
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import windtunnel

windtunnel_object = windtunnel.Windtunnel(None)

stopdeletingmeplease = Axes3D

from custom_color import colormaps

CM = colormaps.ListedColormap(colormaps.viridis.colors[::-1])
# use colormaps.viridis for color maps


PROJECT_PATH = os.path.dirname(windtunnel.__file__)
MODEL_FIG_PATH = os.path.join(PROJECT_PATH, 'data', 'model')
EXPERIMENT_FIG_PATH = os.path.join(PROJECT_PATH, 'data', 'experiments')

X_AXIS_POSITION_LABEL = "Upwind/$x$ (meters)"
Y_AXIS_POSITION_LABEL = "Crosswind/$y$ (meters)"
Z_AXIS_POSITION_LABEL = "Elevation/$z$ (meters)"
PROBABILITY_LABEL = 'Probability'

FIG_FORMAT = ".png"


def get_agent_info(agent_obj):
    if agent_obj is not None:
        titleappend = " cond{condition}|damp{damp}|rF{rf}|wF{wtf}|stmF{stim}|N{total_trajectories}|K{K}|m{m}".format(
            condition=agent_obj.experimental_condition,
            damp=agent_obj.damping_coeff,
            rf=agent_obj.randomF_strength,
            wtf=agent_obj.windF_strength,
            stim=agent_obj.stimF_strength,
            total_trajectories=agent_obj.total_trajectories,
            K=agent_obj.spring_const,
            m=agent_obj.mass
            )
        path = MODEL_FIG_PATH
        is_agent = 'Agent'
    else:
        titleappend = ''
        path = EXPERIMENT_FIG_PATH
        is_agent = 'Mosquito'

    return titleappend, path, is_agent

def draw_cage():
    # makes a little box where the cage is
    # DEPRECIATED in 3D
    cage_midX, cage_midY = 0.1524, 0.  # TODO: turn to absolute boundaries and put in metadata
    return plt.Rectangle((cage_midX - 0.0381, cage_midY - 0.0381), 0.0762, 0.0762,
                         facecolor='none')  # FIXME get rid of hardcoded number


def draw_heater(heater_position, detect_thresh=0.02):  # FIXME detect thresh = 2cm + diam
    """ draws a circle where the heater_position is
    heater_position vector is [x,y, zmin, zmax, diam]
    DEPRECIATED in 3D
    """
    heaterCircle = plt.Circle((heater_position[0], heater_position[1],), heater_position[4] / 2, color='r')
    detectCircle = plt.Circle((heater_position[0], heater_position[1],), detect_thresh, color='gray', fill=False,
                              linestyle='dashed')

    return heaterCircle, detectCircle


def plot_position_heatmaps(trajectories_obj):
    trimmed_df = trajectories_obj._trim_df_endzones()
    ensemble = trimmed_df
    total_trajectories = len(trajectories_obj.get_trajectory_numbers())

    N_points = len(ensemble)
    nbins_x = 100
    nbins_y = 30
    nbins_z = 30

    x = ensemble.position_x.values
    y = ensemble.position_y.values
    z = ensemble.position_z.values

    [x_min, x_max, y_min, y_max, z_min, z_max] = [0.0, 1.0, -0.127, 0.127, 0., 0.254]

    counts, [x_bins, y_bins, z_bins] = np.histogramdd((x, y, z), bins=(nbins_x, nbins_y, nbins_z),
                                                      range=((x_min, x_max), (y_min, y_max), (z_min, z_max)))

    # counts need to be transposed to use pcolormesh
    countsT = counts.T
    probs = countsT / N_points

    # reduce dimensionality for the different plots
    probs_xy = np.sum(probs, axis=0)
    probs_yz = np.sum(probs, axis=2)
    probs_xz = np.sum(probs, axis=1)
    # max_probability = np.max([np.max(probs_xy), np.max(probs_xz), np.max(probs_yz)])
    max_probability = np.max(probs_xy)

    #### file naming and directory selection
    fileappend, path, agent = get_agent_info(trajectories_obj.agent_obj)

    titlebase = "{type} Position Heatmap".format(type=agent)
    numbers = " (n = {})".format(total_trajectories)

    #### XY
    fig0, ax0 = plt.subplots(1)
    heatmap_xy = ax0.pcolormesh(x_bins, y_bins, probs_xy, cmap=CM, vmin=0., vmax=max_probability)
    ax0.set_aspect('equal')
    ax0.invert_yaxis()  # hack to match y axis convention
    # overwrite previous plot schwag
    title1 = titlebase.format(type=type) + " - XY projection" + numbers
    plt.title(title1)
    plt.xlabel(X_AXIS_POSITION_LABEL)
    plt.ylabel(Y_AXIS_POSITION_LABEL)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    the_divider = make_axes_locatable(ax0)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar0 = plt.colorbar(heatmap_xy, cax=color_axis)
    cbar0.ax.set_ylabel(PROBABILITY_LABEL)
    plt.savefig(os.path.join(path, title1 + fileappend + FIG_FORMAT))

    #### XZ
    fig1, ax1 = plt.subplots(1)
    heatmap_xz = ax1.pcolormesh(x_bins, z_bins, probs_xz, cmap=CM, vmin=0., vmax=max_probability)
    ax1.set_aspect('equal')
    plt.xlabel(X_AXIS_POSITION_LABEL)
    plt.ylabel(Z_AXIS_POSITION_LABEL)
    title2 = titlebase.format(type=type) + " - XZ projection" + numbers
    plt.title(title2)
    plt.xlim((x_min, x_max))
    plt.ylim((z_min, z_max))
    the_divider = make_axes_locatable(ax1)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(heatmap_xz, cax=color_axis)
    cbar1.ax.set_ylabel(PROBABILITY_LABEL)
    plt.savefig(os.path.join(path, title2 + fileappend + FIG_FORMAT))


    #### YZ
    fig2, ax2 = plt.subplots(1)
    heatmap_yz = ax2.pcolormesh(y_bins, z_bins, probs_yz, cmap=CM, vmin=0., vmax=max_probability)
    ax2.set_aspect('equal')
    ax2.invert_xaxis()  # hack to match y axis convention
    plt.xlabel(Y_AXIS_POSITION_LABEL)
    plt.ylabel(Z_AXIS_POSITION_LABEL)
    title3 = titlebase.format(type=type) + " - YZ projection" + numbers
    plt.title(title3)
    plt.xlim((y_min, y_max))
    plt.ylim((z_min, z_max))
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(heatmap_yz, cax=color_axis)
    cbar2.ax.set_ylabel(PROBABILITY_LABEL)
    plt.savefig(os.path.join(path, title3 + fileappend + FIG_FORMAT))


    # if windtunnel_object.test_condition is not None:
    #     heaterCircle, detectCircle = draw_heater(windtunnel_object.on_heater_loc)
    #     ax.add_artist(heaterCircle)
    #     ax.add_artist(detectCircle)

    #        #
    #            # draw cage
    #            cage = draw_cage()
    #            hextraj.ax_joint.add_patch(cage)


    #
    # plt.savefig(os.path.join(path, "Trajectories heatmap" + titleappend + FIG_FORMAT))
    # plt.show()


def plot_kinematic_histograms(
        ensemble,
        agent_obj=None,
        plot_kwargs=None,
        titleappend='',
        upw_ensemble="none",
        downw_ensemble="none"):
    statefig = plt.figure()  # , figsize=(9, 8))
    gs1 = gridspec.GridSpec(2, 3)
    axs = [statefig.add_subplot(ss) for ss in gs1]
    # position distributions
    #    pos_all = np.concatenate(pos, axis=0)
    pos_binwidth = .01

    # X pos
    xpos_min, xpos_max = 0.25, .85
    xpos_counts, xpos_bins = np.histogram(ensemble['position_x'],
                                          bins=np.linspace(xpos_min, xpos_max,
                                                           ceil(xpos_max - xpos_min) / pos_binwidth))
    xpos_counts = xpos_counts.astype(float)
    xpos_counts_norm = xpos_counts / xpos_counts.sum()
    axs[0].plot(xpos_bins[:-1] + pos_binwidth / 2, xpos_counts_norm, lw=2, label="Full")
    axs[0].bar(xpos_bins[:-1], xpos_counts_norm, xpos_bins[1] - xpos_bins[0], facecolor='blue', linewidth=0, alpha=0.1)
    if type(downw_ensemble) != str:
        xpos_counts, xpos_bins = np.histogram(downw_ensemble['position_x'],
                                              bins=np.linspace(xpos_min, xpos_max,
                                                               (xpos_max - xpos_min) / pos_binwidth))
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
    axs[0].set_xlabel(Z_AXIS_POSITION_LABEL)
    axs[0].set_ylabel(PROBABILITY_LABEL)

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
    axs[1].set_xlabel(Y_AXIS_POSITION_LABEL)
    axs[1].set_ylabel(PROBABILITY_LABEL)

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
    axs[2].set_xlabel(Z_AXIS_POSITION_LABEL)
    axs[2].set_ylabel(PROBABILITY_LABEL)

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
    axs[3].set_ylabel(PROBABILITY_LABEL, fontsize=12)
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
    axs[4].set_ylabel(PROBABILITY_LABEL)
    axs[4].legend(fontsize=14)

    ## Curvatures

    c_counts, c_bins = np.histogram(ensemble['curvature'])
    c_counts = c_counts.astype(float)
    c_counts_n = c_counts / c_counts.sum()
    axs[5].plot(c_bins[:-1], c_counts_n, label="curvature", lw=2)
    axs[5].fill_between(c_bins[:-1], 0, c_counts_n, facecolor='blue', linewidth=0,
                        alpha=0.1)

    axs[5].set_title("Curvature Distribution")
    axs[5].set_xlabel("Curvature")
    axs[5].set_ylabel(PROBABILITY_LABEL)
    axs[5].legend(fontsize=14)

    ####################

    gs1.tight_layout(statefig, rect=[0, 0.03, 1, 0.95])  # overlapping text hack

    fileappend, path, agent = get_agent_info(agent_obj)

    suptit = "{} Kinematic Distributions".format(agent) + titleappend
    statefig.suptitle(suptit, fontsize=14)

    plt.savefig(os.path.join(path, suptit + FIG_FORMAT))
    plt.show()


#    return xpos_counts_norm, ypos_bins, ypos_counts, ypos_counts_norm, vx_counts_n

def plot_timeseries(ensemble, agent_obj):
    traj_numbers = [int(i) for i in ensemble.index.get_level_values('trajectory_num').unique()]
    data_dict = {}
    times_dict = {}

    for col in ensemble.columns:
        data = []
        col_data = ensemble[col]

        for i in traj_numbers:
            df = col_data.xs(i, level='trajectory_num')
            data.append(df)

        data_dict[col] = data  # every key linked to list of lists

    #### file naming and directory selection
    fileappend, path, agent = get_agent_info(agent_obj)

    titlebase = "{agent} {kinematic} timeseries".format(agent=agent, kinematic="{kinematic}")
    numbers = " (n = {})".format(len(traj_numbers))

    for k, v in iter(sorted(data_dict.iteritems())):
        plt.figure()
        for trial in v:
            plt.plot(trial.index, trial)
        # print v
        # sns.tsplot(data=v, times=times_dict[k], err_style=None) #"unit_traces")
        format_title = titlebase.format(kinematic=k)
        plt.suptitle(format_title + numbers, fontsize=14)
        plt.xlabel("Timestep index")
        plt.ylabel("Value")
        plt.savefig(os.path.join(path, format_title + fileappend + FIG_FORMAT))
        plt.show()


def plot_forces_violinplots(ensemble, agent_obj):
    ensembleF = ensemble.loc[
        (ensemble['position_x'] > 0.25) & (ensemble['position_x'] < 0.95),
            ['totalF_x', 'totalF_y', 'totalF_z',
             'randomF_x', 'randomF_y', 'randomF_z',
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

    fileappend, path, agent = get_agent_info(agent_obj)

    plt.savefig(os.path.join(path, "Force Distributions" + titleappend + FIG_FORMAT))
    plt.show()


def plot_velocity_compassplot(ensemble, agent_obj, kind):
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
            "Agent velocities 0.25 < x < 0.5, center repulsion on (n = {})".format(agent_obj['total_trajectories']),
            y=1.1)

        fileappend, path, agent = get_agent_info(agent_obj)

        plt.savefig(os.path.join(path, "Velocity compass" + fileappend + FIG_FORMAT))
        plt.show()


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


def plot_compass_histogram(vector_name, ensemble, agent_obj, kind='avg_mag_per_bin', title='', fname=''):
    """TODO: make compass plot total mag per bin, overlay compass arrows for avg magnitude

    :param vector_name:
    :param ensemble:
    :param agent_obj:
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
    #        plt.title("Agent velocities 0.25 < x < 0.5, center repulsion on (n = {})".format(agent_obj['total_trajectories']), y=1.1)
    #        plt.savefig("./figs/Compass plot , center repulsion on_ beta{beta}_rf{randomF_scale}_wf{wtf}_N{total_trajectories}.svg".format(\
    #            beta=agent_obj['beta'], randomF_scale=agent_obj['randomF_strength'], wtf=agent_obj['wtF'], total_trajectories=agent_obj['total_trajectories'] + FIG_FORMAT))

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

        fileappend

        plt.savefig(os.path.join(path, "Compass {fname}" + fileappend + FIG_FORMAT))
        plt.show()

        # plt.savefig("./figs/Compass {fname}{append}.svg".format(fname=fname, append=titleappend + FIG_FORMAT))


def draw_cylinder(center_x, center_y, z_min, z_max, r=0.01905, n=5):
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


def plot3D_trajectory(trajectory, plot_kwargs=None):
    '''plotting without coloring
    '''

    fig3D = plt.figure()
    threedee = fig3D.gca(projection='3d')
    # threedee.auto_scale_xyz([0., 1.], [0., 0.3], [0., 0.254])
    # threedee.set_aspect('equal')
    threedee.plot(trajectory.position_y, trajectory.position_x, trajectory.position_z)

    # # Cylinder
    # # get points from cylinder and plot
    # cx, cy, zmin, zmax, diam, label = cylinder(*agent_obj['heater_position'])
    # threedee.plot_wireframe(cx, cy, zmax)

    # Note! I set the y axis to be the X data and vice versa
    threedee.set_ylim(0., 1.)
    threedee.set_xlim(0.127, -0.127)
    threedee.invert_xaxis()  # fix for y convention
    threedee.set_zlim(0., 0.254)
    threedee.set_xlabel("Crosswind/$y$ (meters)", fontsize=14)  # remember! x,y switched in plot() above!
    threedee.set_ylabel("Upwind/$x$ (meters)", fontsize=14)
    threedee.set_zlabel("Elevation/$z$ (meters)", fontsize=14)
    threedee.set_title(plot_kwargs['title'], fontsize=20)

    # plt.savefig("./correlation_figs/Trajectory.svg"., format="svg")

    plt.show()

    # todo: plot cylinder, detect circle, cage


def plot_vector_cloud(trajectories_obj, kinematic, i=None):
    # test whether we are a simulation; if not, forbid plotting of  drivers
    if trajectories_obj.agent_obj is None:
        if kinematic not in ['velocity', 'acceleration']:
            raise TypeError("we don't know the mosquito drivers")


    labels = []
    for dim in ['x', 'y', 'z']:
        labels.append(kinematic + '_' + dim)

    if i is None:
        ensemble = trajectories_obj.data
    else:
        ensemble = trajectories_obj.get_trajectory_i_df(i)

    # grab labels
    vecs = []

    selection = ensemble.loc[:, labels]
    arrays = selection.values
    transpose = arrays.T  # 3 x timesteps matrix
    xs, ys, zs = transpose[0], transpose[1], transpose[2]

    # find longest edge
    longest_edge = abs(max(transpose.min(), transpose.max(), key=abs))
    longest_edge *= 1.05  # add padding

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of {} 3D vector distribution'.format(kinematic))
    ax.autoscale(enable=False, axis='both')  # you will need this line to change the Z-axis
    ax.set_xbound(-longest_edge, +longest_edge)
    ax.set_ybound(-longest_edge, +longest_edge)
    ax.set_zbound(-longest_edge, +longest_edge)

    colors = np.linspace(0, 1, len(xs))
    ensemble_cmap = CM(colors)
    i = range(len(ensemble))

    ax.plot(xs, ys, zs)
    ax.scatter(0, 0, 0, c='r', marker='o', s=50)  # mark origin
    ax.scatter(xs, ys, zs, '.', s=20, linewidths=0, c=ensemble_cmap, alpha=0.2)


def vector_cloud_heatmap(trajectories_obj, kinematic, i=None):
    # test whether we are a simulation; if not, forbid plotting of  drivers
    if trajectories_obj.agent_obj is None:
        if kinematic not in ['velocity', 'acceleration']:
            raise TypeError("we don't know the mosquito drivers")

    labels = []
    for dim in ['x', 'y', 'z']:
        labels.append(kinematic + '_' + dim)

    if i is None:
        ensemble = trajectories_obj.data
    else:
        ensemble = trajectories_obj.get_trajectory_i_df(i)

    # grab labels
    vecs = []

    selection = ensemble.loc[:, labels]
    arrays = selection.values
    transpose = arrays.T  # 3 x timesteps matrix
    xs, ys, zs = transpose[0], transpose[1], transpose[2]

    N_points = len(ensemble)

    counts, [x_bins, y_bins, z_bins] = np.histogramdd((xs, ys, zs), bins=(200, 200, 200))

    # counts need to be transposed to use pcolormesh
    countsT = counts.T
    probs = countsT / N_points

    # reduce dimensionality for the different plots
    probs_xy = np.sum(probs, axis=0)
    probs_yz = np.sum(probs, axis=2)
    probs_xz = np.sum(probs, axis=1)
    # max_probability = np.max([np.max(probs_xy), np.max(probs_xz), np.max(probs_yz)])

    #### XY
    fig0, ax0 = plt.subplots(1)
    heatmap_xy = ax0.pcolormesh(x_bins, y_bins, probs_xy, cmap=CM, vmin=0., )
    plt.scatter(0, 0, c='r', marker='x')
    ax0.set_aspect('equal')
    plt.axis([-1, 1, -1, 1])
    # overwrite previous plot schwag
    the_divider = make_axes_locatable(ax0)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar0 = plt.colorbar(heatmap_xy, cax=color_axis)

    #### XZ
    fig1, ax1 = plt.subplots(1)
    heatmap_xz = ax1.pcolormesh(x_bins, z_bins, probs_xz, cmap=CM, vmin=0.)
    ax1.set_aspect('equal')
    plt.axis([-1, 1, -1, 1])
    plt.scatter(0, 0, c='r', marker='x')
    # plt.xlabel(X_AXIS_POSITION_LABEL)
    # plt.ylabel(Z_AXIS_POSITION_LABEL)
    # title2 = titlebase.format(type=type) + " - XZ projection" + numbers
    # plt.title(title2)
    the_divider = make_axes_locatable(ax1)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(heatmap_xz, cax=color_axis)
    cbar1.ax.set_ylabel(PROBABILITY_LABEL)

    #### YZ
    fig2, ax2 = plt.subplots(1)
    heatmap_yz = ax2.pcolormesh(y_bins, z_bins, probs_yz, cmap=CM, vmin=0.)
    plt.scatter(0, 0, c='r', marker='x')
    plt.axis([-1, 1, -1, 1])
    ax2.set_aspect('equal')
    # plt.xlabel(Y_AXIS_POSITION_LABEL)
    # plt.ylabel(Z_AXIS_POSITION_LABEL)
    # title3 = titlebase.format(type=type) + " - YZ projection" + numbers
    # plt.title(title3)
    the_divider = make_axes_locatable(ax2)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(heatmap_yz, cax=color_axis)
    # cbar2.ax.set_ylabel(PROBABILITY_LABEL)


def vector_cloud_kde(trajectories_obj, kinematic, i=None):
    # test whether we are a simulation; if not, forbid plotting of  drivers
    if trajectories_obj.agent_obj is None:
        if kinematic not in ['velocity', 'acceleration']:
            raise TypeError("we don't know the mosquito drivers")

    labels = []
    for dim in ['x', 'y', 'z']:
        labels.append(kinematic + '_' + dim)

    if i is None:
        ensemble = trajectories_obj.data
    else:
        ensemble = trajectories_obj.get_trajectory_i_df(i)

    # grab labels
    vecs = []

    selection = ensemble.loc[:, labels]

    sns.jointplot(x=labels[0], y=labels[1], data=selection, kind='kde', shade=True,
                  xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), shade_lowest=False, space=0, stat_func=None)

    sns.jointplot(x=labels[0], y=labels[2], data=selection, kind='kde', shade=True,
                  xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), shade_lowest=False, space=0, stat_func=None)

    sns.jointplot(x=labels[1], y=labels[2], data=selection, kind='kde', shade=True,
                  xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), shade_lowest=False, space=0, stat_func=None)


def vector_cloud_pairgrid(trajectories_obj, kinematic, i=None):
    # test whether we are a simulation; if not, forbid plotting of  drivers
    if trajectories_obj.agent_obj is None:
        if kinematic not in ['velocity', 'acceleration']:
            raise TypeError("we don't know the mosquito drivers")

    labels = []
    for dim in ['x', 'y', 'z']:
        labels.append(kinematic + '_' + dim)

    if i is None:
        ensemble = trajectories_obj.data
    else:
        ensemble = trajectories_obj.get_trajectory_i_df(i)

    # grab labels
    vecs = []

    selection = ensemble.loc[:, labels]

    g = sns.PairGrid(selection)
    g.map_upper(plt.scatter, s=2, alpha=0.1)
    g.map_lower(sns.kdeplot, cmap="Blues_d", shade=True)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Pairwise comparison of {agent_type} {kinematic}'.format(agent_type=trajectories_obj.is_agent,
                                                                            kinematic=kinematic))
    g.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8))



    # arrays = selection.values
    # transpose = arrays.T  # 3 x timesteps matrix
    # xs, ys, zs = transpose[0], transpose[1], transpose[2]
    #
    #
    # #### XY
    # fig0, ax0 = plt.subplots(1)
    # heatmap_xy = ax0.pcolormesh(x_bins, y_bins, probs_xy, cmap=CM, vmin=0., vmax=max_probability)
    # plt.scatter(0, 0, c='r', marker='x')
    # ax0.set_aspect('equal')
    # # overwrite previous plot schwag
    # the_divider = make_axes_locatable(ax0)
    # color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    # cbar0 = plt.colorbar(heatmap_xy, cax=color_axis)
    #
    # #### XZ
    # fig1, ax1 = plt.subplots(1)
    # heatmap_xz = ax1.pcolormesh(x_bins, z_bins, probs_xz, cmap=CM, vmin=0., vmax=max_probability)
    # ax1.set_aspect('equal')
    # plt.scatter(0, 0, c='r', marker='x')
    # # plt.xlabel(X_AXIS_POSITION_LABEL)
    # # plt.ylabel(Z_AXIS_POSITION_LABEL)
    # # title2 = titlebase.format(type=type) + " - XZ projection" + numbers
    # # plt.title(title2)
    # the_divider = make_axes_locatable(ax1)
    # color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    # cbar1 = plt.colorbar(heatmap_xz, cax=color_axis)
    # cbar1.ax.set_ylabel(PROBABILITY_LABEL)
    #
    # #### YZ
    # fig2, ax2 = plt.subplots(1)
    # heatmap_yz = ax2.pcolormesh(y_bins, z_bins, probs_yz, cmap=CM, vmin=0., vmax=max_probability)
    # plt.scatter(0, 0, c='r', marker='x')
    # ax2.set_aspect('equal')
    # # plt.xlabel(Y_AXIS_POSITION_LABEL)
    # # plt.ylabel(Z_AXIS_POSITION_LABEL)
    # # title3 = titlebase.format(type=type) + " - YZ projection" + numbers
    # # plt.title(title3)
    # the_divider = make_axes_locatable(ax2)
    # color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    # cbar2 = plt.colorbar(heatmap_yz, cax=color_axis)
    # # cbar2.ax.set_ylabel(PROBABILITY_LABEL)



def plot_all_force_clouds(ensemble):
    import math_sorcery

    # visualize direction selection
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    Npoints = 1000
    data = np.zeros((Npoints, 3))

    for point in range(Npoints):
        data[point] = math_sorcery.gen_symm_vecs(3)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    ax.scatter(x, y, z, c=ensemble_cmap, marker='.', s=80, linewidths=0)
    ax.scatter(0, 0, 0, c='r', marker='o', s=150, linewidths=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_random direction selection")
    plt.savefig('F_rand_direction_selection.png')
    plt.show()




    ##############################################################################################
    # colors for rest of this funct
    ##############################################################################################
    colors = np.linspace(0, 1, len(ensemble))
    ensemble_cmap = CM(colors)
    ##############################################################################################
    # F_total
    ##############################################################################################

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    x = ensemble.totalF_x
    y = ensemble.totalF_y
    z = ensemble.totalF_z

    ax.scatter(x, y, z, marker='.', s=80, linewidths=0, c=ensemble_cmap)
    ax.scatter(0, 0, 0, c='r', marker='o', s=150, linewidths=0)  # mark center

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_total")
    plt.savefig('F_total_cloud.png')
    plt.show()

    ##############################################################################################
    # F_stim
    ##############################################################################################


    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    x = ensemble.stimF_x
    y = ensemble.stimF_y
    z = ensemble.stimF_z

    ax.scatter(x, y, z, marker='.', s=80, linewidths=0, c=ensemble_cmap)
    ax.scatter(0, 0, 0, c='r', marker='o', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_stim")
    plt.savefig('F_stim_cloud.png')
    plt.show()

    ##############################################################################################
    # F_upwind
    ##############################################################################################


    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    x = ensemble.upwindF_x
    y = ensemble.upwindF_y
    z = ensemble.upwindF_z

    ax.scatter(x[:20], y[:20], z[:20], marker='.', s=80, linewidths=0, c=ensemble_cmap)
    ax.scatter(0, 0, 0, c='r', marker='o', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_upwind")
    plt.savefig('F_upwind_cloud.png')
    plt.show()

    ##############################################################################################
    # F_random
    ##############################################################################################

    fig = plt.figure(6)
    ax = fig.add_subplot(111, projection='3d')
    x = ensemble.randomF_x
    y = ensemble.randomF_y
    z = ensemble.randomF_z
    i = range(len(ensemble))

    ax.scatter(x, y, z, marker='.', s=80, linewidths=0, c=ensemble_cmap)
    ax.scatter(0, 0, 0, c='r', marker='o', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_random")
    plt.savefig('F_random_cloud.png')
    plt.show()


    ##############################################################################################
    # F_wall repulsion
    ##############################################################################################

    fig = plt.figure(7)
    ax = fig.add_subplot(111, projection='3d')
    x = ensemble.wallRepulsiveF_x
    y = ensemble.wallRepulsiveF_y
    z = ensemble.wallRepulsiveF_z

    ax.scatter(x, y, z, marker='.', s=80, linewidths=0, c=ensemble_cmap)
    ax.scatter(0, 0, 0, c='r', marker='o', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Visualization of F_wall repulsion")
    plt.savefig('F_wall repulsion_cloud.png')
    plt.show()


def plot_score_comparison(trajectories_obj):
    from scripts.pickle_experiments import load_mosquito_kde_data_dicts

    ref_bins_dict, ref_kde_vals_dict = load_mosquito_kde_data_dicts()
    total_score, scores, targ_kde_vals_dict = trajectories_obj.calc_score()

    for kinematic, ref_vals in ref_kde_vals_dict.iteritems():
        targ_vals = targ_kde_vals_dict[kinematic]

        plt.figure()
        plt.plot(ref_bins_dict[kinematic], targ_vals, label="Input")
        plt.plot(ref_bins_dict[kinematic], ref_vals, label="Reference")
        plt.ylabel("Probability")
        plt.xlabel("Value")
        plt.legend()
        plt.title("Comparison of {kinematic} Probabilities between Control Experiments and Simulation (n = {N})".format(
            kinematic=kinematic, N=len(trajectories_obj.data.index.get_level_values('trajectory_num').unique())))
