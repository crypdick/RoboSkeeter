__author__ = 'richard'

from scipy.stats import entropy
import numpy as np
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
    dkl_v_x = entropy(targ_vals['v_x'], qk=experimental_vals['v_x'])
    dkl_v_y = entropy(targ_vals['v_y'], qk=experimental_vals['v_y'])
    dkl_v_z = entropy(targ_vals['v_z'], qk=experimental_vals['v_z'])
#    dkl_a_x = entropy(targ_vals['a_x'], qk=experimental_vals['a_x'])
#    dkl_a_y = entropy(targ_vals['a_y'], qk=experimental_vals['a_y'])
#    dkl_a_z = entropy(targ_vals['a_z'], qk=experimental_vals['a_z'])
    dkl_c = entropy(targ_vals['c'], qk=experimental_vals['c']) * 3  # scaled up by 6 to increase relative importance

    dkl_scores = [dkl_v_x, dkl_v_y, dkl_v_z, dkl_c]

    # for i, val in enumerate(dkl_scores):
    #     if val > 20:
    #         dkl_scores[i] = 20.

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

    return dkl_score, dkl_scores, targ_vals


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
        bins_dict[k] = np.linspace(0., vect.max() * pad_coeff, 2000)


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
