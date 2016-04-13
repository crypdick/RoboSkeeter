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

import numpy as np
import pandas as pd
from roboskeeter.io import i_o


class Observations(object):
    def __init__(self):
        self.kinematics = pd.DataFrame()

    def concat_df_list(self, dataframe_list):
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

        Output
        ------
        Numbered csvs
        """
        for trajectory_i in self.kinematics.trajectory_num.unique():
            temp_traj = self.get_trajectory_slice(trajectory_i)
            temp_array = temp_traj[['position_x', 'position_y', 'position_z', 'in_plume']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")



    def get_trajectory_slice(self, index=None):
        """
        # TODO: select list of ints
        Parameters
        ----------
        index
            (int or None)
            trajectory index you want to select

        Returns
        -------
        Sliced Pandas df
        """
        if index is None:
            df = self.kinematics
        if type(index) == int:
            df = self.kinematics.loc[self.kinematics.trajectory_num == int(index)]
        else:
            raise ValueError("index must be int or None")

        return df

    def get_trajectory_numbers(self):
        return np.sort(self.kinematics.trajectory_num.unique())

    def experiment_data_to_DF(self, experimental_condition):
        df = i_o.experiment_condition_to_DF(experimental_condition)
        self.kinematics = df

    def _trim_df_endzones(self):
        return self.kinematics.loc[(self.kinematics['position_x'] > 0.05) & (self.kinematics['position_x'] < 0.95)]
