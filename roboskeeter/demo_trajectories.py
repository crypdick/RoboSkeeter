# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:51:37 2015

Demo trajectories

@author: Richard Decal
"""

from matplotlib import pyplot as plt

import generate_trajectory
from trajectory_stats import main as ts

default_wallF = (4e-1, 1e-6, 1e-7, 250)

def demo_single_trajectories():
    print "demo_single_trajectories"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "wall repulsion demo", titleappend = ", normal")
        
    # no forces
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=0., rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", no randomF, wtf only")
    plt.show()
    
    # wallF off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", wallF off")
    plt.show()
    
    # no wind, randomF only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True, v0_stdev=0.01, wtf=0., rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "wall repulsion demo", titleappend = ", no wind, randomF only")
    plt.show()
    
    # no randomF, wtf only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "wall repulsion demo", titleappend = ", no randomF, wtf only")
    plt.show()


def demo_damping():
    print "demo_damping"
    title= "Damping demo"
    # normal
    ts(total_trajectories=10, beta=1e-5, plot_kwargs=\
    {'title':title, 'titleappend':', normal', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()
    
    # normal damping, no driving, high initial velo
    ts(total_trajectories=10, agent_pos="door", target_pos="left", \
        v0_stdev=.04, wtf=0., rf=0., beta=1e-5, Tmax=15, \
        dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None,\
        plot_kwargs={'title':'title', 'titleappend':', normal damp, no driving, high v0', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()
    print "Shows that, absent of driver, velocity decays (therefore damping is working)"
    
    
    # undamped
    ts(total_trajectories=10, agent_pos="door", target_pos="left", \
        v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=0., \
        Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash",\
        wallF=default_wallF, plot_kwargs={'title':'title', 'titleappend':', undamped', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()
    
    # undamped, high rf
    ts(total_trajectories=10, agent_pos="door", target_pos="left",\
        v0_stdev=0.01, wtf=0., rf=4e-05, beta=0., Tmax=15, \
        dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF,\
        plot_kwargs={'title':'title', 'titleappend':', undamped, high rf, wallF normal; wtf off', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()
    
    # normal damp, high rf
    ts(total_trajectories=10, agent_pos="door", target_pos="left", \
        v0_stdev=0.01, wtf=0., rf=4e-05, beta=1e-5, Tmax=15, \
        dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF,\
        plot_kwargs={'title':'title', 'titleappend':', damped, high rf; wallF normal; wtf off', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()
    
    # critical damping
    ts(total_trajectories=10, agent_pos="door", target_pos="left", \
        v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1, Tmax=15, dt=0.01, detect_thresh=0.023175,\
        bounded=True, bounce="crash", wallF=default_wallF, plot_kwargs={'title':'title', 'titleappend':', critical damping', 'singletrajectories':True, 'heatmap':None, 'states':None})
    plt.show()


def demo_randomF_strength():
    print "demo_randomF_strength"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "random_f_strength demo", titleappend = ", normal")
    plt.show()
    
    # off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "random_f_strength demo", titleappend = ", rF off")
    plt.show()
    
    # weak
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "random_f_strength demo", titleappend = ", rF weak")
    plt.show()
    
    # strong
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-05, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "random_f_strength demo", titleappend = ", rF strong")
    plt.show()


def demo_wall_repulsion():
    print "demo_wall_repulsion"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=default_wallF, title = "wall repulsion demo", titleappend = ", normal")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", wall repulsion off")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-5), title = "wall repulsion demo", titleappend = ", wall repulsion weak")
    plt.show()
    
    for i in range(10):
        strongTraj = generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-1), title = "wall repulsion demo", titleappend = ", wall repulsion strong")
    plt.show()
    
    return strongTraj
    
def demo_repulsion_landscape():
    """wallF: 
    wallF_max=8e-8
    decay_const = 90
    mu=0.
    stdev=0.04
    centerF_max=5e-8
    """
    print "demo_repulsion_landscape"

    
    
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF = (8e-08, 90, 0.0, 0.04, 5e-08), title = "wall repulsion demo", titleappend = ", normal")
    plt.show()
    
    # off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", wall repulsion off")
    plt.show()
    
    # weak
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF = (2e-08, 90, 0.0, 0.04, 1e-8), title = "wall repulsion demo", titleappend = ", wall repulsion weak")
    plt.show()
    
    # strong
    for i in range(10):
        strongTraj = generate_trajectory.Trajectory(agent_pos="door", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(8e-07, 90, 0.0, 0.04, 5e-07), title = "wall repulsion demo", titleappend = ", wall repulsion strong")
    plt.show()
    
#demo_single_trajectories()
demo_damping()
#demo_randomF_strength()
#strongTraj = demo_wall_repulsion()
#demo_repulsion_landscape()