__author__ = 'richard'

import agent
import environment
import trajectory


class Base_Experiment(object):
    def __init__(self, **experiment_kwargs):
        for key in experiment_kwargs:
            setattr(self, key, experiment_kwargs[key])

        self.windtunnel = environment.Windtunnel(self.condition)
        self.plume = environment.Plume(self)


class Simulation(Base_Experiment):
    def __init__(self, agent_kwargs, **experiment_kwargs):
        Base_Experiment.__init__(self, **experiment_kwargs)

        self.trajectories = trajectory.Agent_Trajectory(self)
        self.skeeter = agent.Agent(self, **agent_kwargs)
        self.trajectories.add_agent_info(self.skeeter)

        self.skeeter.fly(total_trajectories=self.number_trajectories)


def run_simulation():
    experiment_kwargs = {'initial_position_selection': 'downwind_high',
                         'condition': 'Control',  # {'Left', 'Right', 'Control'}
                         'time_max': 15.,
                         'bounded': True,
                         'number_trajectories': 15
                         }

    agent_kwargs = {'windF_strength': 0.,
                    'randomF_strength': 6.55599224e-06,
                    'stimF_strength': 0.,
                    'spring_const': 0,
                    'damping_coeff': 3.63674551e-07,
                    'collision_type': 'crash',
                    'crash_coeff': 0.8
                    }

    simulation = Simulation(agent_kwargs, **experiment_kwargs)
    # make the skeeter fly. this updates the trajectory_obj
    agent = simulation.skeeter
    trajectory = simulation.skeeter.trajectory_obj
    windtunnel = simulation.windtunnel
    plume = simulation.plume

    print "\nDone."
    return simulation, trajectory, windtunnel, plume, agent


class Experiment(Base_Experiment):
    def __init__(self, experimental_condition):
        Base_Experiment.__init__(self, condition=experimental_condition)
        # self.condition = experimental_condition
        # self.windtunnel = environment.Windtunnel(self.condition)
        # self.plume = environment.Plume(self)
        self.trajectories = trajectory.Experimental_Trajectory()

        self.trajectories.load_experiments(experimental_condition=self.condition)


def get_experiment():
    condition = 'Control'
    experiment = Experiment(condition)
    trajectory, windtunnel, plume = experiment.trajectories, experiment.windtunnel, experiment.plume
    return experiment, trajectory, windtunnel, plume


# simulation, trajectory, windtunnel, plume, agent = run_simulation()

experiment, trajectory, windtunnel, plume = get_experiment()




######################### plotting methods
# simulation.plot_3Dtrajectory(0)  # plot ith trajectory in the ensemble of trajectories

# print pd.isnull(simulation.data).any().nonzero()[0]
# print

# simulation.plot_kinematic_hists()
# simulation.plot_position_heatmaps()
# simulation.plot_force_violin()  # TODO: fix Nans in arrays

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

#
# ######################### manual scoring
# from score import *
# import scripts
#
# experimental_bins, experimental_vals = scripts.pickle_experiments.load_mosquito_kdes()
#
# targ_ensemble = simulation
#
# targ_data = get_data(targ_ensemble)
# targ_KDEs = calc_kde(targ_data)
# targ_vals = evaluate_kdes(targ_KDEs, experimental_bins)  # we evaluate targ KDE at experimental vins for comparison
#
#
# # solve DKL b/w target and reference trajectories
# dkl_v_x = entropy(targ_vals['v_x'], qk=experimental_vals['v_x'])
# dkl_v_y = entropy(targ_vals['v_y'], qk=experimental_vals['v_y'])
# dkl_v_z = entropy(targ_vals['v_z'], qk=experimental_vals['v_z'])
# dkl_a_x = entropy(targ_vals['a_x'], qk=experimental_vals['a_x'])
# dkl_a_y = entropy(targ_vals['a_y'], qk=experimental_vals['a_y'])
# dkl_a_z = entropy(targ_vals['a_z'], qk=experimental_vals['a_z'])
# dkl_c = entropy(targ_vals['c'], qk=experimental_vals['c']) * 6.  # scaled up by 6 to increase relative importance
#
# dkl_scores = [dkl_v_x, dkl_v_y, dkl_v_z, dkl_a_x, dkl_a_y, dkl_a_z, dkl_c]
# for i, val in enumerate(dkl_scores):
#     if val > 20:
#         dkl_scores[i] = 20.
# dkl_score = sum(dkl_scores)
