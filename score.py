__author__ = 'richard'

from scipy.stats import entropy

import scripts.acfanalyze
import scripts.pickle_experiments


def score(agent_traj, experimental_trajs=None):
    if experimental_trajs is None:
        experimental_trajs = scripts.pickle_experiments.load_mosquito_pickle()

    v_bins, a_bins, c_bins = experimental_trajs.experiment_v_bins, experimental_trajs.experiment_a_bins, experimental_trajs.experiment_c_bins
    v_vals_experiment, a_vals_experiment, c_vals_experiment = experimental_trajs.v_vals_experiment, experimental_trajs.a_vals_experiment, experimental_trajs.c_vals_experiment

    v_vals_agent, a_vals_agent, c_vals_agent = agent_traj.evaluate_kernels(v_bins, a_bins, c_bins)


    # solve DKL b/w agent and mosquito experiments
    dkl_v = entropy(v_vals_agent, qk=v_vals_experiment)
    dkl_a = entropy(a_vals_agent, qk=a_vals_experiment)
    dkl_c = entropy(c_vals_agent, qk=c_vals_experiment)

    dkl_scores = [dkl_v, dkl_a, dkl_c]
    dkl_score = sum(dkl_scores)
    # final_score = dkl_v
    # if np.isinf(dkl_score):
    #     dkl_score = 100000

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
