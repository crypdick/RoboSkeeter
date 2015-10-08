__author__ = 'richard'

from scipy.stats import entropy
import numpy as np
from scipy.stats import gaussian_kde as kde

import pandas as pd  # fixme

import scripts.acfanalyze
import scripts.pickle_experiments


def score(targ_ensemble, ref_ensemble='pickle'):
    if ref_ensemble is None:
        ref_ensemble = scripts.pickle_experiments.load_mosquito_trajectory_pickle()  # load control data

        ref_data = get_data(ref_ensemble)
        experimental_bins = calc_bins(ref_data)
        experimental_KDEs = calc_kde(ref_data)
        experimental_vals = evaluate_kdes(experimental_KDEs, experimental_bins)

    if ref_ensemble is 'pickle':
        experimental_bins, experimental_vals = scripts.pickle_experiments.load_mosquito_kdes()

    targ_data = get_data(targ_ensemble)
    targ_KDEs = calc_kde(targ_data, targ_ensemble)
    targ_vals = evaluate_kdes(targ_KDEs, experimental_bins)  # we evaluate targ KDE at experimental vins for comparison


    # solve DKL b/w target and reference trajectories
    dkl_v_x = entropy(targ_vals['v_x'], qk=experimental_vals['v_x'])
    dkl_v_y = entropy(targ_vals['v_y'], qk=experimental_vals['v_y'])
    dkl_v_z = entropy(targ_vals['v_z'], qk=experimental_vals['v_z'])
    dkl_a_x = entropy(targ_vals['a_x'], qk=experimental_vals['a_x'])
    dkl_a_y = entropy(targ_vals['a_y'], qk=experimental_vals['a_y'])
    dkl_a_z = entropy(targ_vals['a_z'], qk=experimental_vals['a_z'])
    dkl_c = entropy(targ_vals['c'], qk=experimental_vals['c']) * 6  # scaled up by 6 to increase relative importance

    dkl_scores = [dkl_v_x, dkl_v_y, dkl_v_z, dkl_a_x, dkl_a_y, dkl_a_z, dkl_c]

    for i, val in enumerate(dkl_scores):
        if val > 20:
            dkl_scores[i] = 20.

    dkl_score = sum(dkl_scores)


    # ################ ACF metrics############
    # # TODO: switch to RMSE of ACFs
    # N_TRAJECTORIES = ensemble.trajectory_num.max()
    #
    # acf_distances = []
    # for i in range(N_TRAJECTORIES):
    #     df = ensemble.loc[ensemble['trajectory_num']==i]
    #     acf_threshcross_index = scripts.acfanalyze.index_drop(df, thresh=ACF_THRESH, verbose=False)
    #     if acf_threshcross_index is 'explosion':  # bad trajectory!
    #         acf_distances.append(300)
    #     else:
    #         acf_distances.append(np.mean(acf_threshcross_index))
    #
    # acf_mean = np.mean(acf_distances)
    # rmse_ACF = abs(acf_mean-16.)
    # # acf_score = np.log(acf_score+1)/20.
    # rmse_ACF /= 20.  #shrink influence
    # print "acf score", rmse_ACF
    # print "dkl_score", dkl_score
    # combined_score = dkl_score + rmse_ACF

    return dkl_score, dkl_scores


def calc_kde(data, ensemble):  # fixme remove ensemble after debugging
    try:
        kdes = {'v_x': kde(data['v_x']),
            'v_y': kde(data['v_y']),
                'v_z': kde(data['v_z'])}
    except ValueError:  # ValueError: array must not contain infs or NaNs
        print "Infs, Nans"
        print "velos", data['v_x'], data['v_y'], data['v_z']
        print "accels", data['a_x'], data['a_y'], data['a_z']
        print ensemble.data['trajectory_num']

    try:
        kdes.update({
            'a_x': kde(data['a_x']),
            'a_y': kde(data['a_y']),
            'a_z': kde(data['a_z']),
            'c': kde(data['c']),
        })
    except ValueError:
        print "v_x", ensemble.data['velocity_x'].values
        print "tF_x", ensemble.data['totalF_x'].values
        print "nulls", pd.isnull(ensemble.data).any()  # .nonzero()[0]
        print "df", ensemble.data
    except np.linalg.linalg.LinAlgError as err:
        print "SINGULAR"
        print data['a_x']
        print ensemble


    return kdes


def get_data(trajectory):
    data = {'v_x': np.abs(trajectory.data['velocity_x'].values),
            'v_y': np.abs(trajectory.data['velocity_y'].values),
            'v_z': np.abs(trajectory.data['velocity_z'].values),
            'a_x': np.abs(trajectory.data['acceleration_x'].values),
            'a_y': np.abs(trajectory.data['acceleration_y'].values),
            'a_z': np.abs(trajectory.data['acceleration_z'].values),
            'c': np.abs(trajectory.data['curvature'].values)
            }
    return data


def calc_bins(data):
    bins_dict = {'v_x': np.linspace(0., data['v_x'].max(), 100),
                 'v_y': np.linspace(0., data['v_y'].max(), 100),
                 'v_z': np.linspace(0., data['v_z'].max(), 100),
                 'a_x': np.linspace(0., data['a_x'].max(), 100),
                 'a_y': np.linspace(0., data['a_y'].max(), 100),
                 'a_z': np.linspace(0., data['a_z'].max(), 100),
                 'c': np.linspace(0., data['c'].max(), 1000)  # more granularity for the curvature
                 }

    return bins_dict


def evaluate_kdes(KDE, bins):
    vals = {'v_x': KDE['v_x'](bins['v_x']),
            'v_y': KDE['v_y'](bins['v_y']),
            'v_z': KDE['v_z'](bins['v_z']),
            'a_x': KDE['a_x'](bins['a_x']),
            'a_y': KDE['a_y'](bins['a_y']),
            'a_z': KDE['a_z'](bins['a_z']),
            'c': KDE['c'](bins['c']),
            }

    return vals
