__author__ = 'richard'

import agent3D
import plume3D
import trajectory3D
from scipy.optimize import minimize, fminbound

myplume = plume3D.Plume()

# wrapper func for agent 3D
def wrapper(param):
    scalar = 1e-7

    wallF_params = [scalar]  #(4e-1, 1e-6, 1e-7, 250)

    # temperature plume

    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    myagent = agent3D.Agent(
        trajectories,
        myplume,
        agent_pos="door",
        heater="left",
        v0_stdev=0.01,
        wtf=7e-07,
        biasF_scale=4e-05,
        stimF_str=1e-4,
        beta=1e-5,
        Tmax=15.,
        dt=0.01,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=wallF_params)
    myagent.fly(total_trajectories=70)

    score = error_fxn(trajectories)
    return score

    # TODO make error functions

def error_fxn(traj_obj):
    pass

def main():
    fminbound(wrapper, )
    pass

if __name__ == '__main__':
    main()