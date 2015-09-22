__author__ = 'richard'

import numpy as np
from scipy.stats import entropy
import scripts.acfanalyze


def score(agent_traj, experimental_trajs, ACF_THRESH=0.5):
    v_bins, a_bins = experimental_trajs.calc_kernel_bins
    v_vals_experiment, a_vals_experiment = experimental_trajs.evaluate_kernels(v_bins, a_bins)

    v_vals_agent, a_vals_agent = agent_traj.evaluate_kernels(v_bins, a_bins)


    # solve DKL b/w agent and mosquito experiments
    dkl_v = entropy(v_vals_agent, qk=v_vals_experiment)
    dkl_a = entropy(a_vals_agent, qk=a_vals_experiment)

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
        acf_threshcross_index = scripts.acfanalyze.index_drop(df, thresh=ACF_THRESH, verbose=False)
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