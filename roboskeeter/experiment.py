__author__ = 'richard'
"""
"""

import roboskeeter.math as m
from roboskeeter.math.kinematic_math import DoMath
# from roboskeeter.math.scoring.scoring import Scoring
from simulator import Simulator
from environment import Environment
from flights import Flights
from roboskeeter.plotting.my_plotter import MyPlotter as plttr


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

        self.flights = None
        self.agent = Simulator(self, agent_kwargs)

        # these get mapped to the correct funcs after run() is ran
        self.plt = None
        self.is_scored = False  # toggle to let analysis functions know whether it has been scored
        self.percent_time_in_plume = None
        self.side_ratio_score = None
        self.score = None

    def run(self, n=None):
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
                self.flights = self.agent.fly(n_trajectories=n)
        else:
            self.flights = Flights()
            self.flights.experiment_data_to_DF(experimental_condition=self.experiment_conditions['condition'])
            self.flights.kinematics['inPlume'] = 0  # FIXME add column to DF
            # TODO: rum plume interaction analysis

        # asign alias
        self.plt = plttr(self)  # TODO: takes self, extracts metadata for files and titles, etc

        # run analysis
        dm = DoMath(self)
        self.flights, self.percent_time_in_plume, self.side_ratio_score = dm.flights, dm.percent_time_in_plume, dm.side_ratio_score

    # def calc_score(self):
    #     # TODO test scoring
        # TODO fix downstream bugs
    #     if self.is_scored == False:
    #         S = Scoring()
    #         self.score = S.score(self.flights)
    #         self.is_scored = True


def start_simulation(num_flights, agent_kwargs=None, experiment_conditions=None):
    """
    Fire up RoboSkeeter
    Parameters
    ----------
    num_flights
        (int) number of flights to simulate
    agent_kwargs
        (dict) params for agent
    experiment_conditions
        (dict) params for environment

    Returns
    -------
    experiment object
    """
    if experiment_conditions is None:
        experiment_conditions = {'condition': 'Right',  # {'Left', 'Right', 'Control'}
                                 'time_max': 6.,
                                 'bounded': True,
                                 'plume_model': "Boolean"  # "Boolean", "timeavg", "None", "Unaveraged"
                                 }
    if agent_kwargs is None:
        agent_kwargs = {'is_simulation': True,
                        'random_f_strength': 6.55599224e-06,
                        'stim_f_strength': 5.0e-06,
                        'damping_coeff': 3.63674551e-07,
                        'collision_type': 'part_elastic',  # 'elastic', 'part_elastic'
                        'restitution_coeff': 0.1,  # 0.8
                        'stimulus_memory_n_timesteps': 1,
                        'decision_policy': 'surge',  # 'surge', 'cast', 'castsurge', 'gradient', 'ignore'
                        'initial_position_selection': 'downwind_high',
                        'verbose': True
                        }

    experiment = Experiment(agent_kwargs, experiment_conditions)
    experiment.run(n=num_flights)
    print "\nDone running simulation."

    return experiment


def load_experiment(experiment_conditions=None):
    """
    Load Sharri's experiments into experiment class
    Parameters
    ----------
    experiment_conditions

    Returns
    -------
    experiment class
    """
    if experiment_conditions is None:  # load defaults
        experiment_conditions = {'condition': 'Control',  # {'Left', 'Right', 'Control'}
                                 'plume_model': "None",  # "Boolean" "None, "Timeavg", "Unaveraged"
                                 'time_max': "N/A (experiment)",
                                 'bounded': True,
                                 }

    agent_kwargs = {'is_simulation': False,
                    'random_f_strength': "UNKNOWN",
                    'stim_f_strength': "UNKNOWN",
                    'damping_coeff': "UNKNOWN",
                    'collision_type': "UNKNOWN",
                    'restitution_coeff': "UNKNOWN",
                    'stimulus_memory_n_timesteps': "UNKNOWN",
                    'decision_policy': "UNKNOWN",
                    'initial_position_selection': "UNKNOWN",
                    'verbose': True
                    }

    experiment = Experiment(agent_kwargs, experiment_conditions)
    experiment.run()
    print "\nDone loading experiment."

    return experiment


if __name__ is '__main__':
    experiment = start_simulation(1, None, None)
    # experiment = load_experiment()  # TODO: experiments should use same code as simulation to figure out plume interaction
                                            # march timestep by timestep

    print "\nAliases updated."
    # useful aliases
    agent = experiment.agent
    kinematics = experiment.flights.kinematics
    windtunnel = experiment.environment.windtunnel
    plume = experiment.environment.plume