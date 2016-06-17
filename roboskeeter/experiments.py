__author__ = 'richard'
"""
"""

from roboskeeter.math.kinematic_math import DoMath
from roboskeeter.math.scoring.scoring import Scoring
from roboskeeter.simulator import Simulator
from roboskeeter.environment import Environment
from roboskeeter.observations import Observations
from roboskeeter.plotting.plot_funcs_wrapper import PlotFuncsWrapper
import numpy as np


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

        self.observations = Observations()
        self.agent = Simulator(self, agent_kwargs)

        # these get mapped to the correct funcs after run() is ran
        self.plt = None
        self.is_scored = False  # toggle to let analysis functions know whether it has been scored
        self.percent_time_in_plume = None
        self.side_ratio_score = None
        self.score, self.score_components = None, None

    def run(self, n=None):  # None as default in case we're loading experiments instead of simulating
        """
        Func that either loads experimental data or runs a simulation, depending on whether self.is_simulation is True
        Parameters
        ----------
        n
            (int, optional)
            Number of flights to simulate. If we are just loading files, it will load the entire ensemble.

        Returns
        -------
        None
        """
        if self.is_simulation:
            if type(n) != int:
                raise TypeError("Number of flights must be integer.")
            else:
                self.observations = self.agent.fly(n_trajectories=n)
        else:
            self.observations.experiment_data_to_DF(experimental_condition=self.experiment_conditions['condition'])
            if self.experiment_conditions['optimizing'] is False:  # skip unneccessary computations for optimizer
                print """\nDone loading files. Iterating through flights and presenting plume, making hypothetical
                 decisions using selected decision policy ({})""".format(self.agent.decision_policy)
                n_rows = len(self.observations.kinematics)
                in_plume = np.zeros(n_rows, dtype=bool)
                plume_signal = np.array([None] * n_rows)
                decision = np.array([None] * n_rows)
                for i, row in self.observations.kinematics.iterrows():
                    in_plume[i] = self.environment.plume.check_in_plume_bounds([row.position_x,
                                                                                row.position_y,
                                                                                row.position_z])
                    decision[i], plume_signal[i] = self.agent.decisions.make_decision(in_plume[i], row.velocity_y)

                self.observations.kinematics['in_plume'] = in_plume
                self.observations.kinematics['plume_signal'] = plume_signal
                self.observations.kinematics['decision'] = decision

        # assign alias
        self.plt = PlotFuncsWrapper(self)  # takes self, extracts metadata for files and titles, etc

        # run analysis

        dm = DoMath(self)  # updates kinematics, etc.
        self.observations, self.percent_time_in_plume, self.side_ratio_score = dm.observations, dm.percent_time_in_plume, dm.side_ratio_score

    def calc_score(self, reference_data=None, score_weights = {'velocity_x': 1,
                                'velocity_y': 1,
                                'velocity_z': 1,
                                'acceleration_x': 1,
                                'acceleration_y': 1,
                                'acceleration_z': 1,  # FIXME change these values back
                                'position_x': 1,
                                'position_y': 1,
                                'position_z': 1,
                                'curvature': 3}):
        if self.is_scored is True:
            pass
        else:
            S = Scoring(self, score_weights, reference_data=reference_data)
            self.score, self.score_components = S.score, S.score_components
            self.is_scored = True
            return self.score, self.score_components



def start_simulation(num_flights, agent_kwargs=None, simulation_conditions=None):
    """
    Fire up RoboSkeeter
    Parameters
    ----------
    num_flights
        (int) number of flights to simulate
    agent_kwargs
        (dict) params for agent
    simulation_conditions
        (dict) params for environment

    Returns
    -------
    experiment object
    """
    if simulation_conditions is None:
        simulation_conditions = {'condition': 'Right',  # {'Left', 'Right', 'Control'}
                                 'time_max': 6.,
                                 'bounded': True,
                                 'optimizing': False,
                                 'plume_model': "Timeavg"  # "Boolean", "Timeavg", "None", "Unaveraged"
                                 }
    if agent_kwargs is None:
        agent_kwargs = {'is_simulation': True,
                        'random_f_strength': 6.64725529e-06, #6.55599224e-06,
                        'stim_f_strength': 5.0e-06,
                        'damping_coeff': 3.63417031e-07, # 3.63674551e-07,
                        'collision_type': 'part_elastic',  # 'elastic', 'part_elastic'
                        'restitution_coeff': 9.99023340e-02, #0.1,  # 0.8
                        'stimulus_memory_n_timesteps': 100,
                        'decision_policy': 'gradient',  # 'surge', 'cast', 'castsurge', 'gradient', 'ignore'
                        'initial_position_selection': 'downwind_high',
                        'verbose': True,
                        'optimizing': False
                        }

    experiment = Experiment(agent_kwargs, simulation_conditions)
    experiment.run(n=num_flights)
    if agent_kwargs['verbose'] is True:
        print "\nDone running simulation."

    return experiment


def load_experiment(condition='Control'):
    """
    Load Sharri's experiments into experiment class
    Parameters
    ----------
    experiment_conditions

    Returns
    -------
    experiment class
    """
    experiment_conditions = {'condition': condition,  # {'Left', 'Right', 'Control'}
                             'plume_model': "Boolean",  # "Boolean" "None, "Timeavg", "Unaveraged"
                             'time_max': "N/A (experiment)",
                             'bounded': True,
                             'optimizing': False
                             }

    agent_kwargs = {'is_simulation': False,  # ALL THESE VALUES ARE A HYPOTHESIS!!!
                    'random_f_strength': "UNKNOWN",
                    'stim_f_strength': "UNKNOWN",
                    'damping_coeff': "UNKNOWN",
                    'collision_type': "UNKNOWN",
                    'restitution_coeff': "UNKNOWN",
                    'stimulus_memory_n_timesteps': "UNKNOWN",
                    'decision_policy': "surge",  # 'surge', 'cast', 'castsurge', 'gradient', 'ignore'
                    'initial_position_selection': "UNKNOWN",
                    'verbose': True
                    }

    experiment = Experiment(agent_kwargs, experiment_conditions)
    experiment.run()
    print "\nDone loading experiment."

    return experiment


if __name__ is '__main__':
    # experiment = start_simulation(200, None, None)
    experiment = load_experiment('Left')

    print "\nAliases updated."
    # useful aliases
    agent = experiment.agent
    kinematics = experiment.observations.kinematics
    windtunnel = experiment.environment.windtunnel
    plume = experiment.environment.plume
    plotter = experiment.plt