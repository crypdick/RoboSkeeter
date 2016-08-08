__author__ = 'richard'

from scipy.stats import ks_2samp
from roboskeeter.experiments import load_experiment


class Scoring():
    def __init__(self,
                 condition='Control',
                 score_weights=None,
                 reference_data=None
                 ):
        self.condition = condition
        print "Scoring reference ensemble set to {}".format(self.condition)
        if reference_data is None:
            # TODO: when called from an experiment class, find out the relevant experiment from metadata,
            # load that experiment, use as reference
            print "no reference data provided; loading experimental data"
            self.reference_data = self._load_reference_ensemble(self.condition)
        else:  # if ensemble provided
            self.reference_data = reference_data

        if score_weights is None:  # no score weights provided
            print "No score weights provided, loading defaults"
            self.score_weights = {'velocity_x': 1,
                'velocity_y': 1,
                'velocity_z': 1,
                'acceleration_x': 1,
                'acceleration_y': 1,
                'acceleration_z': 1,
                'position_x': 1,
                'position_y': 1,
                'position_z': 1,
                'curvature': 3}
        else:
            self.score_weights = score_weights

    def calc_score(self, target_experiment):
        """
        Solves KS-2 sample test for test and reference data, weighted by the score weights kwarg

        Returns
        -------
        total score and score components
        """
        target_data = target_experiment.observations.get_kinematic_dict(trim_endzones=True)

        score_components = dict()
        for kinematic, kinematic_array in target_data.iteritems():
            ks_score, pval = ks_2samp(kinematic_array, self.reference_data[kinematic])
            # cubing scores <1 will make them smaller. +1 makes sure they will increase exponentially
            score_plus_1_cubed = (ks_score+1)**3
            score_components[kinematic] = self.score_weights[kinematic] * score_plus_1_cubed

        return sum(score_components.values()), score_components

    def _load_reference_ensemble(self, condition):
        """
        stores trimmed experimental data
        Parameters
        ----------
        condition

        Returns
        -------

        """
        reference_experiment = load_experiment([condition])
        return reference_experiment.observations.get_kinematic_dict(trim_endzones=True)
