__author__ = 'richard'

import os
import numpy as np
import pandas as pd
from glob import glob
import statsmodels.tsa
import statsmodels.graphics.tsaplots
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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
        'pos_x',
        'pos_y',
        'pos_z',
        'velo_x',
        'velo_y',
        'velo_z',
        'accel_x',
        'accel_y',
        'accel_z',
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


csv_list = make_csv_name_list()

for csv_name in csv_list:
    df = load_trajectory_dynamics_csv(csv_name)
    print csv_name, 'size/timesteps = ', df.size

    if not os.path.exists('./correlation_figs/{data_name}'.format(data_name = csv_name)):
        os.makedirs('./correlation_figs/{data_name}'.format(data_name = csv_name))

    for label, col in df.iteritems():
        if label in ['velo_x', 'velo_y', 'velo_z']:
            acf = statsmodels.tsa.stattools.acf(col, nlags = 70)
            print label, arg_less(acf, .5)