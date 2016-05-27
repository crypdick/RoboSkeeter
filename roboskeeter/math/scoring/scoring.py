__author__ = 'richard'

import numpy as np
from scipy.stats import ks_2samp



class Scoring():
    def __init__(self,
                 target_experiment,
                 reference_data=None,
                 scoring_kwargs={'kinematics_list' : ['velocities', 'curviture'], 'dimensions': ['x', 'y', 'z']},
                 score_weights={'velocity_x': 1,
                                'velocity_y': 1,
                                'velocity_z': 1,
                                'acceleration_x': 0,
                                'acceleration_y': 0,
                                'acceleration_z': 0,
                                'position_x': 0,
                                'position_y': 0,
                                'position_z': 0,
                                'curvature': 3}
                 ):
        if reference_data is None:
            # when called from an experiment class, find out the relevant experiment from metadata, load that experiment, use as reference
            reference_experiment = self.load_reference_ensemble(target_experiment.experiment_conditions['condition'])
            self.reference_data = reference_experiment.observations.get_kinematic_dict()
        else:
            self.reference_data = reference_data

        self.score_weights = score_weights

        self.target_data = target_experiment.observations.get_kinematic_dict()

        self.score, self.score_components = self.calc_score()

    def calc_score(self):
        """
        Solves KS-2 sample test for test and reference data, weighted by the score weights kwarg

        Returns
        -------
        total score and score components
        """
        score_components = dict()
        for k, v in self.target_data.iteritems():
            ks, pval = ks_2samp(v, self.reference_data[k])
            score_components[k] = self.score_weights[k] * ks

        return sum(score_components.values()), score_components

    def load_reference_ensemble(self, condition):
        from roboskeeter import experiments
        reference_experiment = experiments.load_experiment(experiment_conditions = {'condition': condition,
                                 'plume_model': "None",  # "Boolean" "None, "Timeavg", "Unaveraged"
                                 'time_max': "N/A (experiment)",
                                 'bounded': True,
                                 })
        return reference_experiment
