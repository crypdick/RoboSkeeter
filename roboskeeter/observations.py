# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

creates a trajectory object that has a bunch of sweet methods.
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
        """we don't use self.observations.to_csv(name) because Sharri likes having separate csvs for each trajectory

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
        if type(index) == int or type(index) == np.int64:
            df = self.kinematics.loc[self.kinematics.trajectory_num == int(index)]
        else:
            raise ValueError("index must be int or None, found type {} instead".format(type(index)))

        return df

    def get_trajectory_numbers(self):
        return np.sort(self.kinematics.trajectory_num.unique())

    def experiment_data_to_DF(self, experimental_condition):
        """
        for loading experimental data to df
        Parameters
        ----------
        experimental_condition

        Returns
        -------

        """
        try:
            df = i_o.experiment_condition_to_DF(experimental_condition)
            self.kinematics = df
        except TypeError:
            raise AssertionError('Input experimental condition should be string, instead got {}. printed: {}'.format(
                type(experimental_condition), experimental_condition))
        except IOError:
            print """IOerror. You are probably missing the trajectory data in /data/trajectories. The data can be found at
                https://drive.google.com/open?id=0B1CyEg2BqCdjY2p5SXRuQTcwNmc
                Note, the data is encrypted until we publish. If you're a collaborator, email Richard Decal for the password."""
            raise

    def _trim_df_endzones(self):
        return self.kinematics.loc[(self.kinematics['position_x'] > 0.05) & (self.kinematics['position_x'] < 0.95)]

    def get_kinematic_dict(self, trim_endzones = False):
        if trim_endzones:
            kinematics = self._trim_df_endzones()
        else:
            kinematics = self.kinematics
        dict = {'velocity_x': kinematics['velocity_x'].values,
               'velocity_y': kinematics['velocity_y'].values,
               'velocity_z': kinematics['velocity_z'].values,
               'position_x': kinematics['position_x'].values,
               'position_y': kinematics['position_y'].values,
               'position_z': kinematics['position_z'].values,
               'acceleration_x': kinematics['acceleration_x'].values,
               'acceleration_y': kinematics['acceleration_y'].values,
               'acceleration_z': kinematics['acceleration_z'].values,
               'curvature': kinematics['curvature'].values
               }

        return dict

    def get_starting_positions(self):
        positions_at_timestep_0 = self.kinematics.loc[(self.kinematics.tsi == 0), ['position_x', 'position_y', 'position_z']]
        return positions_at_timestep_0


"""
the following is the code I used to fit the intiial velocity using the control experimental flight data

v0 = kinematics.loc[(kinematics.tsi == 0),['velocity_x','velocity_y','velocity_z']]
v0_norm = np.linalg.norm(v0, axis=1)
mu, std = scipy.stats.norm.fit(v0_norm)

"""