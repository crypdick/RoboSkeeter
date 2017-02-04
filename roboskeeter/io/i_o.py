# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:42 2015

@author: richard
"""
import os
import string
from Tkinter import Tk
from tkFileDialog import askdirectory

import numpy as np
import pandas as pd


def load_single_csv_to_df(csv_dir):
    """

    Parameters
    ----------
    csv_dir
        str, full directory of csv

    Returns
    -------
    pandas df
    """

    col_labels = [  # TODO: check that Sharri's kinematics are the same as your kinematics
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

    # map string "NaN" to np.nan
    # header=None is needed to make sure dtype float is assigned properly, apparently
    dataframe = pd.read_csv(csv_dir, na_values="NaN", names=col_labels, header=None, dtype=np.float32)

    dataframe.fillna(value=0, inplace=True)  # TODO: there shouldn't be NaNs in data

    df_len = len(dataframe.position_x)

    # take fname number
    fname = os.path.split(csv_dir)[1]
    fname_num = extract_number_from_fname(fname)

    dataframe['trajectory_num'] = [fname_num] * df_len
    dataframe['tsi'] = np.arange(df_len)
    # plume related stuff will get set inside Experiment()

    return dataframe


def experiment_condition_to_DF(experimental_condition):
    """
    Given an experimental condition, load the appropriate dataset into a dataframe.
    Parameters
    ----------
    experimental_condition
        (string)
        Control, Left, or Right, or list thereof

    Returns
    -------
    df
    """
    if type(experimental_condition) is str:
        experimental_condition = [experimental_condition]
    experimental_condition = [string.upper(i) for i in experimental_condition]
    df_list = []
    for condition in experimental_condition:
        dir_label = "EXP_TRAJECTORIES_" + condition
        directory = get_directory(dir_label)

        for fname in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:  # list files
            print "Loading {} from {}".format(fname, directory)
            file_path = os.path.join(directory, fname)
            dataframe = load_single_csv_to_df(file_path)

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
    dirname = os.path.dirname
    PROJECT_PATH = dirname(dirname(dirname(__file__)))
    EXPERIMENTS_PATH = os.path.join(PROJECT_PATH, 'data', 'experiments')
    MODEL_PATH = os.path.join(PROJECT_PATH, 'data', 'model')

    EXPERIMENTAL_TRAJECTORIES = os.path.join(EXPERIMENTS_PATH, 'trajectories')
    EXP_TRAJECTORIES_CONTROL = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'control')
    EXP_TRAJECTORIES_LEFT = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'left')
    EXP_TRAJECTORIES_RIGHT = os.path.join(EXPERIMENTAL_TRAJECTORIES, 'right')

    TEMPERATURES_PATH = os.path.join(EXPERIMENTS_PATH, 'temperature')
    RAW = os.path.join(TEMPERATURES_PATH, 'raw-data')
    TIMEAVG = os.path.join(TEMPERATURES_PATH, 'timeavg-data')
    VAR = os.path.join(TEMPERATURES_PATH, 'variance-model')
    BOOL = os.path.join(TEMPERATURES_PATH, 'boolean-model')
    VAR_LEFT_CSV = os.path.join(VAR, 'left', 'LeftplumeVar_nonan.csv')
    VAR_RIGHT_CSV = os.path.join(VAR, 'right', 'RightplumeVar_nonan.csv')
    THERMOCOUPLE_RAW_LEFT_CSV = os.path.join(RAW, 'left', 'raw_left.csv')
    THERMOCOUPLE_RAW_RIGHT_CSV = os.path.join(RAW, 'right', 'raw_right.csv')
    THERMOCOUPLE_TIMEAVG_LEFT_CSV = os.path.join(TIMEAVG, 'left', 'timeavg_left.csv')
    THERMOCOUPLE_TIMEAVG_LEFT_PADDED_CSV = os.path.join(TIMEAVG, 'left', 'timeavg_left_padded.csv')
    THERMOCOUPLE_TIMEAVG_LEFT_INTERPOLATED_CSV = os.path.join(TIMEAVG, 'left', 'timeavg_left_interpolated.csv')
    THERMOCOUPLE_TIMEAVG_RIGHT_CSV = os.path.join(TIMEAVG, 'right', 'timeavg_right.csv')
    THERMOCOUPLE_TIMEAVG_RIGHT_PADDED_CSV = os.path.join(TIMEAVG, 'right', 'timeavg_right_padded.csv')
    THERMOCOUPLE_TIMEAVG_RIGHT_INTERPOLATED_CSV = os.path.join(TIMEAVG, 'right', 'timeavg_right_interpolated.csv')
    BOOL_LEFT_CSV = os.path.join(BOOL, 'left', 'left_plume_bounds.csv')
    BOOL_RIGHT_CSV = os.path.join(BOOL, 'right', 'right_plume_bounds.csv')




    dirs = {
        'PROJECT_PATH': PROJECT_PATH,
        'MODEL_PATH': MODEL_PATH,
        'EXPERIMENT_PATH': EXPERIMENTS_PATH,
        'EXPERIMENTAL_TRAJECTORIES': EXPERIMENTAL_TRAJECTORIES,
        'EXP_TRAJECTORIES_CONTROL': EXP_TRAJECTORIES_CONTROL,
        'EXP_TRAJECTORIES_LEFT': EXP_TRAJECTORIES_LEFT,
        'EXP_TRAJECTORIES_RIGHT': EXP_TRAJECTORIES_RIGHT,
        'THERMOCOUPLE_RAW_LEFT': THERMOCOUPLE_RAW_LEFT_CSV,
        'THERMOCOUPLE_TIMEAVG_LEFT_PADDED_CSV': THERMOCOUPLE_TIMEAVG_LEFT_PADDED_CSV,
        'THERMOCOUPLE_TIMEAVG_LEFT_INTERPOLATED_CSV': THERMOCOUPLE_TIMEAVG_LEFT_INTERPOLATED_CSV,
        'THERMOCOUPLE_RAW_RIGHT': THERMOCOUPLE_RAW_RIGHT_CSV,
        'THERMOCOUPLE_TIMEAVG_RIGHT_PADDED_CSV': THERMOCOUPLE_TIMEAVG_RIGHT_PADDED_CSV,
        'THERMOCOUPLE_TIMEAVG_RIGHT_INTERPOLATED_CSV': THERMOCOUPLE_TIMEAVG_RIGHT_INTERPOLATED_CSV,
        'THERMOCOUPLE_TIMEAVG_LEFT_CSV': THERMOCOUPLE_TIMEAVG_LEFT_CSV,
        'THERMOCOUPLE_TIMEAVG_RIGHT_CSV': THERMOCOUPLE_TIMEAVG_RIGHT_CSV,
        'BOOL_LEFT_CSV': BOOL_LEFT_CSV,
        'BOOL_RIGHT_CSV': BOOL_RIGHT_CSV,
        'VAR_LEFT_CSV': VAR_LEFT_CSV,
        'VAR_RIGHT_CSV': VAR_RIGHT_CSV
    }

    if selection is None:
        print("Enter directory with experimental data")
        Tk().withdraw()
        directory = askdirectory()
    else:
        directory = dirs[selection]

    print("Directory selected: {}".format(directory))

    return directory


def extract_number_from_fname(token):
    extract_digits = lambda stng: "".join(char for char in stng if char in string.digits)
    to_int = lambda x: int(float(x)) if x.count(".") <= 1 else None

    number = to_int(extract_digits(token))

    return number
