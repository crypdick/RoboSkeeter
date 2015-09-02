__author__ = 'richard'

import os
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.stattools import acf
# import statsmodels.graphics.tsaplots
import matplotlib.pyplot as plt

plt.style.use('ggplot')

ACF_THRESH = 0.5
TRAJECTORY_DATA_DIR = "experimental_data/control_trajectories/"

def make_csv_name_list():
    # TODO export this to io
    
    print "Loading + filtering CSV files from ", TRAJECTORY_DATA_DIR
    os.chdir(TRAJECTORY_DATA_DIR)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__))  # go back to old dir

    return csv_list

def load_trajectory_dynamics_csv(data_fname):
    file_path = os.path.join(os.getcwd(), TRAJECTORY_DATA_DIR, data_fname + ".csv")

    col_labels = [
        'position_x',
        'position_y',
        'position_z',
        'velocity_x',
        'velocity_y',
        'velocity_z',
        'acceleration_x',
        'acceleration_y',
        'acceleration_z',
        'heading_angle',
        'angular_velo_xy',
        'angular_velo_yz',
        'curvature'
    ]

    dyn_trajectory_DF = pd.read_csv(file_path, na_values="NaN", names=col_labels)  # recognize string as NaN
    dyn_trajectory_DF.fillna(value=0, inplace=True)


    return dyn_trajectory_DF


def arg_less(inarray,threshold):
    filtered = np.nonzero(inarray<threshold)
    return np.nonzero(inarray<threshold)[0][0]  # return index of first item that is under thresh


def index_drop(df, thresh=ACF_THRESH, verbose=True):
    """given df, thresh, return at what index the acf dipped below thresh"""
    if verbose is True:
        print 'size/timesteps = ', df.size

    indices = []
    for label, col in df.iteritems():
        if label in ['velocity_x', 'velocity_y', 'velocity_z']:
            ACF = acf(col, nlags = 70)
            if verbose is True:
                print label, arg_less(ACF, ACF_THRESH)
            indices.append(arg_less(ACF, ACF_THRESH))

    return indices

if __name__ == '__main__':
    csv_list = make_csv_name_list()

    for csv_name in csv_list:
        print csv_name
        df = load_trajectory_dynamics_csv(csv_name)
        scores = index_drop(df, ACF_THRESH, verbose=False)
        print scores