# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:42 2015

@author: richard
"""
import os
from Tkinter import Tk
from tkFileDialog import askdirectory
import numpy as np
import pandas as pd
import string
import unit_tests  # hack to get root dir


def load_csv_to_df(data_fname, rel_dir ="data/processed_trajectories/"):
    """

    Parameters
    ----------
    data_fname
        name of csv file
    rel_dir
        dir where the file lives

    Returns
    -------
    pandas df
    """
    file_path = os.path.join(os.getcwd(), rel_dir, data_fname + ".csv")

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

    dataframe = pd.read_csv(file_path, na_values="NaN", names=col_labels)  # recognize str(NaN) as NaN
    dataframe.fillna(value=0, inplace=True)

    return dataframe


def experiment_data_to_DF(experimental_condition):
    """

    Parameters
    ----------
    experimental_condition
        (string)
        Control, Left, or Right

    Returns
    -------
    df
    """
    condition = string.capitalize(experimental_condition)
    dir_label = "EXP_TRAJECTORIES_" + condition
    directory = get_directory(dir_label)

    df_list = []

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
        'heading_angleS',
        'angular_velo_xyS',
        'angular_velo_yzS',
        'curvatureS'
    ]

    for fname in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:  # list files
        print "Loading {} from {}".format(fname, directory)
        file_path = os.path.join(directory, fname)
        # base_name = os.path.splitext(file_path)[0]

        dataframe = pd.read_csv(file_path, na_values="NaN", names=col_labels)  # recognize str(NaN) as NaN
        dataframe.fillna(value=0, inplace=True)

        df_len = len(dataframe.position_x)

        # take fname number (slice out "control_" and ".csv")
        fname_num = int(fname[:-4])

        # ensure that csv fname is just an int
        if type(fname_num) is not int:
            raise ValueError('''we are expecting csv files to be integers.
                instead, we found type {}. culprit: {}'''.format(type(fname_num), fname))

        dataframe['trajectory_num'] = [fname_num] * df_len
        dataframe['tsi'] = np.arange(df_len)

        df_list.append(dataframe)

    df = pd.concat(df_list)
    return df


def get_csv_name_list(path, relative=True):
    if relative:
        return os.listdir(os.path.join(os.path.realpath('.'), path))
    else:
        return os.listdir(path)


def get_csv_filepath_list(path, csv_list):
    """

    Parameters
    ----------
    path
    csv_list

    Returns
    -------

    """
    paths = [os.path.join(path, fname) for fname in csv_list]
    return paths


def get_directory(selection=None):
    """Centralized func to define directories, or select using dialog box

    In:
    Selection
        None, open dialog box
        PROJECT_PATH = os.path.dirname(trajectory.__file__)
        MODEL_PATH = os.path.join(PROJECT_PATH, 'data', 'model')
        EXPERIMENT_PATH = os.path.join(PROJECT_PATH, 'data', 'experiments')
        CONTROL_EXP_PATH = os.path.join(EXPERIMENT_PATH, 'control_processed_and_filtered')

    Out:
    directory path
    """
    PROJECT_PATH = os.path.dirname(unit_tests.__file__)
    EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'data', 'experiments')
    MODEL_PATH = os.path.join(PROJECT_PATH, 'data', 'model')

    EXPERIMENTAL_TRAJECTORIES = os.path.join(EXPERIMENTS_PATH, 'trajectories')
    EXP_TRAJECTORIES_CONTROL = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'control')
    EXP_TRAJECTORIES_LEFT = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'left')
    EXP_TRAJECTORIES_RIGHT = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'right')

    PLUME_PATH = os.path.join(EXPERIMENTS_PATH, 'plume_data')
    THERMOCOUPLE = os.path.join(PLUME_PATH, 'thermocouple')
    THERMOCOUPLE_RAW_LEFT = os.path.join(THERMOCOUPLE, 'raw_left')
    THERMOCOUPLE_RAW_RIGHT = os.path.join(THERMOCOUPLE, 'raw_right')
    THERMOCOUPLE_TIMEAVG_LEFT_CSV = os.path.join(THERMOCOUPLE, 'timeavg', 'left', 'timeavg_left.csv')
    THERMOCOUPLE_TIMEAVG_RIGHT_CSV = os.path.join(THERMOCOUPLE, 'timeavg', 'right', 'timeavg_right.csv')
    BOOL_LEFT_CSV = os.path.join(PLUME_PATH, 'boolean', 'left', 'left_plume_bounds.csv')
    BOOL_RIGHT_CSV = os.path.join(PLUME_PATH, 'boolean', 'right', 'right_plume_bounds.csv')




    dirs = {
        'PROJECT_PATH': PROJECT_PATH,
        'MODEL_PATH': MODEL_PATH,
        'EXPERIMENT_PATH': EXPERIMENTS_PATH,
        'EXPERIMENTAL_TRAJECTORIES': EXPERIMENTAL_TRAJECTORIES,
        'EXP_TRAJECTORIES_CONTROL': EXP_TRAJECTORIES_CONTROL,
        'EXP_TRAJECTORIES_LEFT': EXP_TRAJECTORIES_LEFT,
        'EXP_TRAJECTORIES_RIGHT': EXP_TRAJECTORIES_RIGHT,
        'THERMOCOUPLE': THERMOCOUPLE,
        'THERMOCOUPLE_RAW_LEFT': THERMOCOUPLE_RAW_LEFT,
        'THERMOCOUPLE_RAW_RIGHT': THERMOCOUPLE_RAW_RIGHT,
        'THERMOCOUPLE_TIMEAVG_LEFT_CSV': THERMOCOUPLE_TIMEAVG_LEFT_CSV,
        'THERMOCOUPLE_TIMEAVG_RIGHT_CSV': THERMOCOUPLE_TIMEAVG_RIGHT_CSV,
        'BOOL_LEFT_CSV': BOOL_LEFT_CSV,
        'BOOL_RIGHT_CSV': BOOL_RIGHT_CSV,
    }


    if selection is None:
        print("Enter directory with experimental data")
        Tk().withdraw()
        directory = askdirectory()
    else:
        directory = dirs[selection]

    print("Directory selected: {}".format(directory))

    return directory


def extract_number_from_fname(self, token):
    extract_digits = lambda stng: "".join(char for char in stng if char in string.digits + ".")
    to_float = lambda x: float(x) if x.count(".") <= 1 else None

    try:
        return to_float(extract_digits(token))
    except ValueError:
        print token, extract_digits(token)


if __name__ == '__main__':
    a = load_csv_to_df('Control-27')
