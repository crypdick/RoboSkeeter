__author__ = 'richard'

import numpy as np
from scipy.stats import entropy
# from scipy.stats import gaussian_kde as kde
import scripts.pickle_experiments
from scripts.math_sorcery import calculate_1Dkde, evaluate_kde


class Scoring():
    def __init__(self, ref_bins_vals='pickle'):

        if ref_bins_vals is 'pickle':  # load already calculated KDEs
            print "loading re-calculated scoring data as reference. make sure this pickle is up-to-date!"
            self.experimental_bins, self.experimental_vals = scripts.pickle_experiments.load_mosquito_kde_data_dicts()
        else:  # provided
            self.experimental_bins, self.experimental_vals = ref_bins_vals

        self.targ_data, self.targ_KDEs, self.targ_vals = None, None, None
        self.dkls, self.dkl_sum = None, None

    def score_ensemble(self, ensemble, kinematics_list=['velocities', 'curvature']):
        self.targ_vals = self._get_target_data(ensemble)
        self.dkls = self._calc_dkls(kinematics_list)
        self.dkl_sum = sum(self.dkls)

        return self.dkl_sum

    def _get_target_data(self, ensemble):
        targ_data = get_data(ensemble)
        targ_KDEs = calc_kde(targ_data)
        targ_vals = evaluate_kdes(targ_KDEs,
                                  self.experimental_bins)  # we evaluate targ KDE at experimental bins for comparison

        return targ_vals

    def _calc_dkls(self, kinematics_list):
        kinematic_dict = {'velocities': ['v_x', 'v_y', 'v_z'],
                          'accelerations': ['a_x', 'a_y', 'a_z'],
                          'curvature': ['c'],
                          'positions': ['p_x', 'p_y', 'p_z'],
                          'crosswind_position': ['p_y']}

        kinematic_tokens = []
        for kin in kinematics_list:
            kinematic_tokens.extend(kinematic_dict[kin])

        dkls = []
        for token in kinematic_tokens:
            exp_distribution = self.experimental_vals[token]
            targ_distribution = self.targ_vals[token]
            dkls.append(entropy(targ_distribution, qk=exp_distribution))

        return dkls




def get_data(trajectory):
    data = {'v_x': trajectory.data['velocity_x'].values,
            'v_y': trajectory.data['velocity_y'].values,
            'v_z': trajectory.data['velocity_z'].values,
#            'a_x': np.abs(trajectory.data['acceleration_x'].values),
#            'a_y': np.abs(trajectory.data['acceleration_y'].values),
#            'a_z': np.abs(trajectory.data['acceleration_z'].values),
            'c': trajectory.data['curvature'].values
            }
    return data


def calc_bins(data):
    pad_coeff = 2.  # pad the distribution to properly penalize values above
    bins_dict = {}
    for k, vect in data.items():
        lower, upper = vect.min(), vect.max()
        if lower > 0:
            lower = 0.
        bins_dict[k] = np.linspace(lower * pad_coeff, upper * pad_coeff, 2000)


    return bins_dict


def calc_kde(data):
    '''takes a dictionary of vectors and returns a kernel density function of each vector '''
    kdes_dict = {}

    for k, v in data.items():
        kdes_dict[k] = calculate_1Dkde(v)

    return kdes_dict


def evaluate_kdes(kdes_dict, bins_dict):
    '''given a KDE function and positions, it will evaluate the KDE function at those locations '''
    vals_dict = {}

    for kinem, kde in kdes_dict.items():
        bins = bins_dict[kinem]
        vals_dict[kinem] = evaluate_kde(kde, bins)

    return vals_dict
