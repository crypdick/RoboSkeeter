__author__ = 'richard'

import environment
import trajectory
from agent import Agent


class Base_Experiment(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.windtunnel = environment.Windtunnel(self.condition)

        if self.plume_type == "Boolean":
            self.plume = environment.Boolean_Plume(self)
        elif self.plume_type == "timeavg":
            self.plume = environment.Timeavg_Plume(self)
        elif self.plume_type == "None":
            self.plume = None
        else:
            raise NotImplementedError("no such plume type {}".format(self.plume_type))


class Simulation(Base_Experiment):
    def __init__(self, agent_kwargs, **experiment_kwargs):
        super(self.__class__, self).__init__(**experiment_kwargs)

        self.trajectories = trajectory.Agent_Trajectory(self)
        self.skeeter = Agent(self, agent_kwargs)
        self.trajectories.add_agent_info(self.skeeter)

        self.skeeter.fly(total_trajectories=self.number_trajectories)


class Experiment(Base_Experiment):
    def __init__(self, **experiment_kwargs):
        super(self.__class__, self).__init__(**experiment_kwargs)

        self.trajectories = trajectory.Experimental_Trajectory(self)
        self.trajectories.load_experiments(experimental_condition=self.condition)

def run_simulation(agent_kwargs, experiment_kwargs):
    if experiment_kwargs is None:
        experiment_kwargs = {'condition': 'Right',  # {'Left', 'Right', 'Control'}
                             'time_max': 6.,
                             'bounded': True,
                             'number_trajectories': 1,
                             'plume_type': "timeavg"  # "Boolean", "timeavg"
                             }
    if agent_kwargs is None:
        agent_kwargs = {'randomF_strength': 6.55599224e-06,
                        'stimF_strength': 5.0e-06,
                        'damping_coeff': 3.63674551e-07,
                        'collision_type': 'part_elastic',  # 'elastic', 'part_elastic'
                        'restitution_coeff': 0.1,  # 0.8
                        'stimulus_memory': 1,
                        'decision_policy': 'gradient',  # 'surge_only', 'cast_only', 'cast+surge', 'gradient', 'ignore'
                        'initial_position_selection': 'downwind_high',
                        'verbose': True
                        }

    simulation = Simulation(agent_kwargs, **experiment_kwargs)
    # make the skeeter fly. this updates the trajectory_obj
    agent = simulation.skeeter
    trajectory = simulation.skeeter.trajectory_obj
    windtunnel = simulation.windtunnel
    plume = simulation.plume

    print "\nDone."
    return simulation, trajectory, windtunnel, plume, agent


def get_experiment():
    # pick which data to retrieve
    experiment_kwargs = {'condition': 'Control',  # {'Left', 'Right', 'Control'}
                         'plume_type': "None"#"Boolean" "None, "
                         }

    experiment = Experiment(**experiment_kwargs)
    trajectory, windtunnel, plume = experiment.trajectories, experiment.windtunnel, experiment.plume
    return experiment, trajectory, windtunnel, plume


if __name__ is '__main__':
    simulation, trajectory_s, windtunnel, plume, agent = run_simulation(None, None)
    # experiment, trajectory_e, windtunnel, plume = get_experiment()


######################### dump data for csv for Sharri
# print "dumping to csvs"
# e = simulation.data
# r = e.trajectory_num.iloc[-1]
#
# for i in range(r+1):
#     e1 = e.loc[e['trajectory_num'] == i, ['position_x', 'position_y', 'position_z', 'inPlume']]
#     e1.to_csv("l"+str(i)+'.csv', index=False)
#
# # simulation.plot_kinematic_hists()
# # simulation.plot_posheatmap()
# # simulation.plot_force_violin()
#
# # # for plume stats
# # g = e.loc[e['behavior_state'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]

# experiments
# [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 40, 41, 43, 44, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
