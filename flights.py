# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

creates a trajectory object that has a bunch of sweet methods.
takes vectors from agent and adds it to ensemble

trajectory.flights
trajectory.append_ensemble(a trajectory)

trajectory.describe()
    agent info
    plots

trajectory.save
"""
import os
import string
from analysis.kinematic_math import DoMath
import numpy as np
import pandas as pd


from scripts import i_o, animate_trajectory_callable


class Flights(object):
    def __init__(self):
        self.kinematics = pd.DataFrame()
        self.percent_time_in_plume = None

    def load_observations_and_analyze(self, dataframe_list):
        """
        Takes list of pandas dataframes, concatinates them, and runs analysis functions.
        Parameters
        ----------
        dataframe_list:
            list of pd dataframes to be concatinated and analyzed

        Returns
        -------
        None
        """
        self.kinematics = pd.concat(dataframe_list)


    def dump2csvs(self):
        """we don't use self.flights.to_csv(name) because Sharri likes having separate csvs for each trajectory
        """
        for trajectory_i in self.kinematics.trajectory_num.unique():
            temp_traj = self.get_trajectory_i_df(trajectory_i)
            temp_array = temp_traj[['position_x', 'position_y']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")

    def calc_side_ratio_score(self):
        """upwind left vs right ratio"""  # TODO replace with KF score
        upwind_half = self.kinematics.loc[self.kinematics.position_x > 0.5]
        total_pts = len(upwind_half)

        left_upwind_pts = len(self.kinematics.loc[(self.kinematics.position_x > 0.5) & (self.kinematics.position_y < 0)])
        right_upwind_pts = total_pts - left_upwind_pts
        print "seconds extra on left side: ", (left_upwind_pts - right_upwind_pts) / 100.
        ratio = float(left_upwind_pts) / right_upwind_pts
        print ratio

    def _extract_number_from_fname(self, token):
        extract_digits = lambda stng: "".join(char for char in stng if char in string.digits + ".")
        to_float = lambda x: float(x) if x.count(".") <= 1 else None

        try:
            return to_float(extract_digits(token))
        except ValueError:
            print token, extract_digits(token)

    def get_trajectory_i_df(self, index):
        return self.kinematics.loc[self.kinematics.trajectory_num == int(index)]


    def get_trajectory_numbers(self):
        return np.sort(self.kinematics.trajectory_num.unique())


    def load_experiments(self, experimental_condition):
        dir_labels = {
            'Control': 'EXP_TRAJECTORIES_CONTROL',
            'Left': 'EXP_TRAJECTORIES_LEFT',
            'Right': 'EXP_TRAJECTORIES_RIGHT'}
        directory = i_o.get_directory(dir_labels[experimental_condition])

        self.agent_obj = None

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

        self.load_observations_and_analyze(dataframe_list=df_list)

    def _trim_df_endzones(self):
        return self.kinematics.loc[(self.kinematics['position_x'] > 0.05) & (self.kinematics['position_x'] < 0.95)]
