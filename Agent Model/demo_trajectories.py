# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:51:37 2015

Demo trajectories

@author: Richard Decal
"""

import generate_trajectory
from matplotlib import pyplot as plt

def demo_single_trajectories():
    print "demo_single_trajectories"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "wall repulsion demo", titleappend = ", normal")
        
    # no forces
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=0., rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", no randomF, wtf only")
    plt.show()
    
    # wallF off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", wallF off")
    plt.show()
    
    # no wind, randomF only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True, v0_stdev=0.01, wtf=0., rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "wall repulsion demo", titleappend = ", no wind, randomF only")
    plt.show()
    
    # no randomF, wtf only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "wall repulsion demo", titleappend = ", no randomF, wtf only")
    plt.show()


def demo_damping():
    print "demo_damping"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "Damping demo", titleappend = ", normal")
    plt.show()
    
    # normal damping, no driving, high initial velo
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.4, wtf=0., rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "Damping demo", titleappend = ", normal damp, no driving, high v0")
    plt.show()
    print "Shows that, absent of driver, velocity decays (therefore damping is working)"
    
    
    # undamped
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=0., Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "Damping demo", titleappend = ", undamped")
    plt.show()
    
    # undamped, high rf
    for i in range(2):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=0., rf=4e-05, beta=0., Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "Damping demo", titleappend = ", undamped, high rf, wallF & wtf off")
    plt.show()
    
    # normal damp, high rf
    for i in range(2):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=0., rf=4e-05, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "Damping demo", titleappend = ", damped, high rf, wallF & wtf off")
    plt.show()
    
    # critical damping
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "Damping demo", titleappend = ", critical damping ")
    plt.show()


def demo_randomF_strength():
    print "demo_randomF_strength"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "randomF_strength demo", titleappend = ", normal")
    plt.show()
    
    # off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "randomF_strength demo", titleappend = ", rF off")
    plt.show()
    
    # weak
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "randomF_strength demo", titleappend = ", rF weak")
    plt.show()
    
    # strong
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-05, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "randomF_strength demo", titleappend = ", rF strong")
    plt.show()


def demo_wall_repulsion():
    print "demo_wall_repulsion"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), title = "wall repulsion demo", titleappend = ", normal")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, title = "wall repulsion demo", titleappend = ", wall repulsion off")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-5), title = "wall repulsion demo", titleappend = ", wall repulsion weak")
    plt.show()
    
    for i in range(10):
        strongTraj = generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-1), title = "wall repulsion demo", titleappend = ", wall repulsion strong")
    plt.show()
    
    return strongTraj
    
    
#demo_single_trajectories()
#demo_damping()
#demo_randomF_strength()
strongTraj = demo_wall_repulsion()