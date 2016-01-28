__author__ = 'richard'
import cPickle as pickle
import os

from scripts import i_o
import trajectory
import experiment
import score

reload(score)
reload(trajectory)

EXPERIMENT_PATH = i_o.get_directory('EXPERIMENT_PATH')


def store_mosquito_pickle():
    exper = experiment.Experiment(**{'condition': 'Control', 'plume_type': "None"})
    pickle.dump(exper, open(os.path.join(EXPERIMENT_PATH, "controls.p"), "wb"))

    ref_data = score.get_data(exper.trajectories)
    experimental_bins_dict = score.calc_bins(ref_data)
    experimental_KDEs_dict = score.calc_kde(ref_data)
    experimental_vals_dict = score.evaluate_kdes(experimental_KDEs_dict, experimental_bins_dict)

    pickle.dump(experimental_bins_dict, open(os.path.join(EXPERIMENT_PATH, "experimental_bins_dict.p"), "wb"))
    pickle.dump(experimental_vals_dict, open(os.path.join(EXPERIMENT_PATH, "experimental_vals_dict.p"), "wb"))
    print "Pickles dumped."


def load_mosquito_trajectory_pickle():
    return pickle.load(open(os.path.join(EXPERIMENT_PATH, "controls.p"), "rb"))


def load_mosquito_kde_data_dicts():
    bins = pickle.load(open(os.path.join(EXPERIMENT_PATH, "experimental_bins_dict.p"), "rb"))
    vals = pickle.load(open(os.path.join(EXPERIMENT_PATH, "experimental_vals_dict.p"), "rb"))

    return bins, vals


if __name__ is '__main__':
    store_mosquito_pickle()
