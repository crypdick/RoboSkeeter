__author__ = 'richard'

import numpy as np
from scipy.stats import entropy
# from scipy.stats import gaussian_kde as kde
import scripts.acfanalyze
import scripts.pickle_experiments
from scripts.math_sorcery import calculate_1Dkde, evaluate_kde


def score(targ_ensemble, ref_bins_vals='pickle'):
    if ref_bins_vals is 'pickle':  # load already calculated KDEs
        experimental_bins, experimental_vals = scripts.pickle_experiments.load_mosquito_kde_data_dicts()
    else:  # provided
        experimental_bins, experimental_vals = ref_bins_vals

    targ_data = get_data(targ_ensemble)
    targ_KDEs = calc_kde(targ_data)
    targ_vals = evaluate_kdes(targ_KDEs, experimental_bins)  # we evaluate targ KDE at experimental bins for comparison

    # solve DKL b/w target and reference trajectories
    dkl_v_y = entropy(targ_vals['v_y'], qk=experimental_vals['v_y'])
    dkl_p_y = entropy(targ_vals['p_y'], qk=experimental_vals['p_y'])

    dkl_scores = [dkl_p_y, dkl_v_y]
    dkl_sum = sum(dkl_scores) * 10

    return dkl_sum, dkl_scores, targ_vals


def get_data(trajectory):
    data = {'p_y': trajectory.data['position_y'].values,
            'v_y': trajectory.data['velocity_y'].values,
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
    kdes_dict = {}

    for k, v in data.items():
        kdes_dict[k] = calculate_1Dkde(v)

    return kdes_dict


def evaluate_kdes(kdes_dict, bins_dict):
    vals_dict = {}

    for kinem, kde in kdes_dict.items():
        bins = bins_dict[kinem]
        vals_dict[kinem] = evaluate_kde(kde, bins)

    return vals_dict
