__author__ = 'richard'

import numpy as np

from statsmodels.tsa.stattools import acf

# import statsmodels.graphics.tsaplots
import matplotlib.pyplot as plt
import scripts.data_io as data_io
import trajectory

plt.style.use('ggplot')

ACF_THRESH = 0.5
NLAGS = 70
TRAJECTORY_DATA_REL_DIR = "experiments/control_trajectories/"

csv_list = data_io.make_csv_name_list(TRAJECTORY_DATA_REL_DIR)


def arg_less(inarray,threshold):
    filtered = np.nonzero(inarray<threshold)  # False is interpreted as 0
    try:
        return filtered[0][0]  # return index of first item that is under thresh
    except IndexError:  # occurs when never goes under thresh
        return NLAGS


def index_drop(df, thresh=ACF_THRESH, verbose=True):
    """given df, thresh, return at what index the acf dipped below thresh"""
    if verbose is True:
        print 'size/timesteps = ', df.size

    indices = []
    for label, col in df.iteritems():
        if label in ['velocity_x', 'velocity_y', 'velocity_z']:
            try:
                ACF = acf(col, nlags = NLAGS)
            except TypeError:  # happens when ACF can't be done b/c there are fewer than nlags lags
                return 'explosion'
            if verbose is True:
                print label, arg_less(ACF, ACF_THRESH)
            indices.append(arg_less(ACF, ACF_THRESH))

    return indices

if __name__ == '__main__':
    csv_list = data_io.make_csv_name_list(TRAJECTORY_DATA_REL_DIR)

    df_list = []
    traj_count = 0
    for csv_name in csv_list:
        # print csv_name
        df = data_io.load_csv2DF(csv_name, rel_dir=TRAJECTORY_DATA_REL_DIR)
        df['trajectory_num'] = traj_count
        df_list.append(df)
        traj_count += 1
        scores = index_drop(df, ACF_THRESH, verbose=False)
        # print scores

    trajectories_object = trajectory.Trajectory()
    trajectories_object.load_ensemble(df_list)