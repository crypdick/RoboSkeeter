__author__ = 'richard'

import numpy as np
from scipy.stats import ks_2samp



class Scoring():
    def __init__(self,
                 target_experiment,
                 reference_experiment=None,
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
        if reference_experiment is None:
            # when called from an experiment class, find out the relevant experiment from metadata, load that experiment, use as reference
            reference_experiment = self.load_reference_ensemble(target_experiment.experiment_conditions['condition'])

        self.score_weights = score_weights

        self.target_data, self.reference_data = self._select_data(target_experiment, reference_experiment)

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

    def _select_data(self, target_experiment, reference_experiment):
        # TODO: make depend on self.scoring_kwargs
        r_k = reference_experiment.observations.kinematics
        t_k = target_experiment.observations.kinematics


        reference_data = {'velocity_x': r_k['velocity_x'].values,
                          'velocity_y': r_k['velocity_y'].values,
                          'velocity_z': r_k['velocity_z'].values,
                          'position_x': r_k['position_x'].values,
                          'position_y': r_k['position_y'].values,
                          'position_z': r_k['position_z'].values,
                          'acceleration_x': np.abs(r_k['acceleration_x'].values),
                          'acceleration_y': np.abs(r_k['acceleration_y'].values),
                          'acceleration_z': np.abs(r_k['acceleration_z'].values),
                          'curvature': r_k['curvature'].values
                          }

        target_data =    {'velocity_x': t_k['velocity_x'].values,
                          'velocity_y': t_k['velocity_y'].values,
                          'velocity_z': t_k['velocity_z'].values,
                          'position_x': t_k['position_x'].values,
                          'position_y': t_k['position_y'].values,
                          'position_z': t_k['position_z'].values,
                          'acceleration_x': np.abs(t_k['acceleration_x'].values),
                          'acceleration_y': np.abs(t_k['acceleration_y'].values),
                          'acceleration_z': np.abs(t_k['acceleration_z'].values),
                          'curvature': t_k['curvature'].values
                          }
        return target_data, reference_data
