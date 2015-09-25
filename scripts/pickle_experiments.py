__author__ = 'richard'
import os
import pickle

import trajectory

PROJECT_PATH = os.path.dirname(trajectory.__file__)
MODEL_PATH = os.path.join(PROJECT_PATH, 'data', 'model')
EXPERIMENT_PATH = os.path.join(PROJECT_PATH, 'data', 'experiments')


def store_mosquito_pickle():
    MOSQUITOES = trajectory.Trajectory()
    MOSQUITOES.load_experiments()
    pickle.dump(MOSQUITOES, open(os.path.join(EXPERIMENT_PATH, "controls.p"), "wb"))


def load_mosquito_pickle():
    return pickle.load(open(os.path.join(EXPERIMENT_PATH, "controls.p"), "rb"))
