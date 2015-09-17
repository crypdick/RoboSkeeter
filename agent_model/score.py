__author__ = 'richard'

import numpy as np
from scipy.stats import entropy
import acfanalyze
from trajectory3D import Trajectory

# load experimentally observed velo + accel distributions
experimental_trajs = Trajectory()
experimental_trajs.load_ensemble('experiments')

experimental_trajs.calc_kinematic_kernels()
v_experimental_pos, a_experimental_pos = experimental_trajs.kde_v_positions, experimental_trajs.kde_a_positions
# eval at positions
v_experimental_vals, a_experimental_vals = experimental_trajs.evaluate_kernels(positions=[v_experimental_pos, a_experimental_pos])


def score(ensemble, ACF_THRESH=0.5):
    # get histogram vals for agent ensemble
    _, v_test_vals, _, a_test_vals = experimental_trajs.calc_kinematic_kernels(positions=(v_experimental_pos,a_experimental_pos))


    # turn into prob dist
    a_counts = a_counts.astype(float)
    a_total_counts = a_counts.sum()
    a_counts_n = a_counts / a_total_counts
    # print a_counts_n

    # solve DKL
    dkl_a = entropy(a_counts_n, qk=a_dist_observed)

    v_binwidth =  0.015
    vmin, vmax = 0., 0.605
    v_3Dvect_magnitude = ensemble['velocity_3Dmagn'].values
    v_counts, v_bins = np.histogram(v_3Dvect_magnitude, bins=np.arange(vmin, vmax, v_binwidth))
    v_counts = v_counts.astype(float)
    v_total_counts = v_counts.sum()
    # turn into prob dist
    v_counts_n = v_counts / v_total_counts  # todo make sure this is true

    # solve DKL
    dkl_v = entropy(v_counts_n, qk=v_dist_observed)
    # print 'dkl_v' , dkl_v


    dkl_scores = [dkl_v, dkl_a]
    dkl_score = sum(dkl_scores)
    # final_score = dkl_v
    # if np.isinf(dkl_score):
    #     dkl_score = 100000

    ################ ACF metrics############
    # TODO: switch to RMSE of ACFs
    N_TRAJECTORIES = ensemble.trajectory_num.max()

    acf_distances = []
    for i in range(N_TRAJECTORIES):
        df = ensemble.loc[ensemble['trajectory_num']==i]
        acf_threshcross_index = acfanalyze.index_drop(df, thresh=ACF_THRESH, verbose=False)
        if acf_threshcross_index is 'explosion':  # bad trajectory!
            acf_distances.append(300)
        else:
            acf_distances.append(np.mean(acf_threshcross_index))

    acf_mean = np.mean(acf_distances)
    rmse_ACF = abs(acf_mean-16.)
    # acf_score = np.log(acf_score+1)/20.
    rmse_ACF /= 20.  #shrink influence
    print "acf score", rmse_ACF
    print "dkl_score", dkl_score
    combined_score = dkl_score + rmse_ACF

    if np.isnan(combined_score):
        combined_score = 1e10

    return combined_score, dkl_scores, dkl_a, dkl_v, rmse_ACF, a_bins, a_counts_n, v_bins, v_counts_n