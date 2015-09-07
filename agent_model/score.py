__author__ = 'richard'

import numpy as np
from scipy.stats import entropy
import acfanalyze

# load csv values
a_csv = np.genfromtxt('experimental_data/acceleration_distributions_uw.csv',delimiter=',')
a_csv = a_csv.T
a_observed = a_csv[4][:-1]  # throw out last datum

v_csv = np.genfromtxt('experimental_data/velocity_distributions_uw.csv',delimiter=',')
v_csv = v_csv.T
v_observed = v_csv[4][:-1]  # throw out last datum

def score(ensemble, ACF_THRESH):
    # get histogram vals for agent ensemble
    adist = 0.1
    # |a|
    amin, amax = 0., 4.9+adist  # arange drops last number, so pad range by dist
    # pdb.set
    accel_all_magn = ensemble['acceleration_3Dmagn'].values
    a_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.arange(amin, amax, adist))
    # turn into prob dist
    a_counts = a_counts.astype(float)
    a_total_counts = a_counts.sum()
    a_counts_n = a_counts / a_total_counts
    # print a_counts_n

    # solve DKL
    dkl_a = entropy(a_counts_n, qk=a_observed)

    vdist =  0.015
    vmin, vmax = 0., 0.605
    velo_all_magn = ensemble['velocity_3Dmagn'].values
    v_counts, vabs_bins = np.histogram(velo_all_magn, bins=np.arange(vmin, vmax, vdist))
    v_counts = v_counts.astype(float)
    v_total_counts = v_counts.sum()
    # turn into prob dist
    v_counts_n = v_counts / v_total_counts

    # solve DKL
    dkl_v = entropy(v_counts_n, qk=v_observed)
    # print 'dkl_v' , dkl_v


    dkl_scores = [dkl_a, dkl_v]
    dkl_score = sum(dkl_scores)
    # final_score = dkl_v
    # if np.isinf(dkl_score):
    #     dkl_score = 100000

    ################ ACF metrics############
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
    acf_score = abs(acf_mean-16.)
    # print "mean", acf_mean
    # acf_score = np.log(acf_score+1)/20.
    acf_score /= 20.  #shrink influence
    print "acf score", acf_score
    print "dkl_score", dkl_score
    combined_score = dkl_score + acf_score

    print "final_score", combined_score

    if np.isnan(combined_score):
        combined_score = 1e10

    return combined_score, dkl_scores, dkl_a, dkl_v, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n