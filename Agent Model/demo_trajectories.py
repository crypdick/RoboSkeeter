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
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", normal")
        
    # no forces
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=0., rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, titleappend = ", no randomF, wtf only")
    plt.show()
    
    # wallF off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, titleappend = ", wallF off")
    plt.show()
    
    # no wind, randomF only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True, v0_stdev=0.01, wtf=0., rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", no wind, randomF only")
    plt.show()
    
    # no randomF, wtf only
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0., beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", no randomF, wtf only")
    plt.show()


def demo_damping():
    print "demo_damping"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", normal")
    plt.show()
    
    # undamped
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=0., Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", undamped")
    plt.show()
    
    # critical damping
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", critical damping ")
    plt.show()


def demo_randomF_strength():
    print "demo_randomF_strength"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", normal")
    plt.show()
    
    # off
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=0, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", rF off")
    plt.show()
    
    # weak
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", rF weak")
    plt.show()
    
    # strong
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=1e-05, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", rF strong")
    plt.show()


def demo_wall_repulsion():
    print "demo_wall_repulsion"
    # normal
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4), titleappend = ", normal")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=None, titleappend = ", wall repulsion off")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-5), titleappend = ", wall repulsion weak")
    plt.show()
    
    for i in range(10):
        generate_trajectory.Trajectory(agent_pos="cage", target_pos="left", plotting = True,   v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-1), titleappend = ", wall repulsion strong")
    plt.show()
    
    
#demo_single_trajectories()
demo_damping()
#demo_randomF_strength()
#demo_wall_repulsion()