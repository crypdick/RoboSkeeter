__author__ = 'richard'
import pickle
import os

from scripts import i_o
import trajectory

CONTROLS = i_o.get_directory('CONTROL_EXP_PATH')
EXPERIMENT_PATH = i_o.get_directory('EXPERIMENT_PATH')


def store_mosquito_pickle():
    MOSQUITOES = trajectory.Experimental_Trajectory()
    MOSQUITOES.load_experiments(directory=CONTROLS)
    pickle.dump(MOSQUITOES, open(os.path.join(EXPERIMENT_PATH, "controls.p"), "wb"))


def load_mosquito_pickle():
    return pickle.load(open(os.path.join(EXPERIMENT_PATH, "controls.p"), "rb"))
