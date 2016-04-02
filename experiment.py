__author__ = 'richard'
"""
"""

from environment import Environment
from kinematics import Kinematics
from agent import Agent
import analysis


class Experiment(object):
    """
    Experiment object which both experiments and real experiments share in common.

    Stores the windtunnel and plume objects
    """
    def __init__(self, agent_kwargs, experiment_conditions):
        # save metadata
        self.experiment_conditions = experiment_conditions
        self.is_simulation = agent_kwargs['is_simulation']

        # init objects
        self.environment = Environment(self)
        self.kinematics = Kinematics(self)
        self.agent = Agent(self, agent_kwargs)

        # these get mapped to the correct funcs after run() is ran
        self.analysis = None
        self.scoring = None


    def run(self, N=None):
        if self.is_simulation:
            if type(N) != int:
                raise TypeError("Number of flights must be integer.")
            else:
                self.agent.fly(total_trajectories=N)
        else:
            self.kinematics.load_experiments(experimental_condition=self.condition)

        self.analysis = analysis
        self.scoring = analysis.scoring





def start_simulation(N_flights, agent_kwargs=None, experiment_conditions=None):
    if experiment_conditions is None:
        experiment_conditions = {'condition': 'Right',  # {'Left', 'Right', 'Control'}
                                 'time_max': 6.,
                                 'bounded': True,
                                'plume_model': "timeavg"  # "Boolean", "timeavg"
                                 }
    if agent_kwargs is None:
        agent_kwargs = {'is_simulation': True,
                        'randomF_strength': 6.55599224e-06,
                        'stimF_strength': 5.0e-06,
                        'damping_coeff': 3.63674551e-07,
                        'collision_type': 'part_elastic',  # 'elastic', 'part_elastic'
                        'restitution_coeff': 0.1,  # 0.8
                        'stimulus_memory_N_timesteps': 1,
                        'decision_policy': 'gradient',  # 'surge_only', 'cast_only', 'cast+surge', 'gradient', 'ignore'
                        'initial_position_selection': 'downwind_high',
                        'verbose': True
                        }

    experiment = Experiment(agent_kwargs, experiment_conditions)
    experiment.run(N = N_flights)
    print "\nDone running simulation."

    return experiment


def load_experiment():
    # pick which data to retrieve
    experiment_conditions = {'condition': 'Control',  # {'Left', 'Right', 'Control'}
                         'plume_model': "None", #"Boolean" "None, "Timeavg",
                         'time_max': "N/A (experiment)",
                         'bounded': True,
                         }

    agent_kwargs = {'is_simulation': False,
                        'randomF_strength': "UNKNOWN",
                        'stimF_strength': "UNKNOWN",
                        'damping_coeff': "UNKNOWN",
                        'collision_type': "UNKNOWN",
                        'restitution_coeff': "UNKNOWN",
                        'stimulus_memory_N_timesteps': "UNKNOWN",
                        'decision_policy': "UNKNOWN",
                        'initial_position_selection': "UNKNOWN",
                        'verbose': True
                        }

    experiment = Experiment(agent_kwargs, experiment_conditions)
    print "\nDone loading experiment."

    return experiment


if __name__ is '__main__':
    experiment = start_simulation(1, None, None)

    agent = experiment.agent
    kinematics = experiment.agent.kinematics
    windtunnel = experiment.environment.windtunnel
    plume = experiment.environment.plume


######################### dump data for csv for Sharri
# print "dumping to csvs"
# e = experiment.data
# r = e.trajectory_num.iloc[-1]
#
# for i in range(r+1):
#     e1 = e.loc[e['trajectory_num'] == i, ['position_x', 'position_y', 'position_z', 'inPlume']]
#     e1.to_csv("l"+str(i)+'.csv', index=False)
#
# # experiment.plot_kinematic_hists()
# # experiment.plot_posheatmap()
# # experiment.plot_force_violin()
#
# # # for plume stats
# # g = e.loc[e['behavior_state'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]