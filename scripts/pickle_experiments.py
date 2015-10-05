__author__ = 'richard'
import pickle
import os

from scripts import i_o
import trajectory
import score

EXPERIMENT_PATH = i_o.get_directory('EXPERIMENT_PATH')


def store_mosquito_pickle():
    MOSQUITOES = trajectory.Experimental_Trajectory()
    MOSQUITOES.load_experiments(selection='CONTROL_EXP_PATH')  # load the experimental data
    pickle.dump(MOSQUITOES, open(os.path.join(EXPERIMENT_PATH, "controls.p"), "wb"))
    ref_data = score.get_data(MOSQUITOES)
    experimental_bins = score.calc_bins(ref_data)
    experimental_KDEs = score.calc_kde(ref_data)
    experimental_vals = score.evaluate_kdes(experimental_KDEs, experimental_bins)

    pickle.dump(experimental_bins, open(os.path.join(EXPERIMENT_PATH, "experimental_bins.p"), "wb"))
    pickle.dump(experimental_vals, open(os.path.join(EXPERIMENT_PATH, "experimental_vals.p"), "wb"))


def load_mosquito_trajectory_pickle():
    return pickle.load(open(os.path.join(EXPERIMENT_PATH, "controls.p"), "rb"))


def load_mosquito_kdes():
    bins = pickle.load(open(os.path.join(EXPERIMENT_PATH, "experimental_bins.p"), "rb"))
    vals = pickle.load(open(os.path.join(EXPERIMENT_PATH, "experimental_vals.p"), "rb"))

    return bins, vals
