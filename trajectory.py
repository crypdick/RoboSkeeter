# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

creates a trajectory object that has a bunch of sweet methods.
takes vectors from agent and adds it to ensemble

trajectory.data
trajectory.append_ensemble(a trajectory)

trajectory.describe()
    agent info
    plots

trajectory.save
"""
import numpy as np
import os
import string

import pandas as pd

from scripts import i_o
from scripts import plotting
from scripts.math_sorcery import calculate_curvature, calculate_xy_heading_angle, calculate_xy_magnitude
import score


#def T_find_stats(t_targfinds):
#    """Target finding stats. Returns average time to find the target, and the
#    number of successes
#    
#    TODO:  find tfind stats for ensemble
#    """
#    t_targfinds = np.array(t_targfinds)
#    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
#    if len(t_finds_NoNaNs) == 0:
#
#        return np.nan  # no target finds, no stats to run
#    else:
#        num_success = float(len(t_finds_NoNaNs))
#        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
#
#        return Tfind_avg


class Trajectory():
    def __init__(self):
        self.reload_data()
        self.agent_obj = None
        self.velo_kernel = None
        self.accel_kernel = None
        self.curvature_kernel = None
        self.is_agent = None

    def reload_data(self):
        self.data = pd.DataFrame()

    def load_ensemble_and_analyze(self, data):
        if type(data) is dict:
            trajectory = pd.DataFrame(data)
            self.data.append(trajectory)  # this should be avoided b/c lots of overhead (slow)
            self.run_analysis()
        elif type(data) is list:
            self.data = pd.concat(data)  # fast
            try:
                self.run_analysis()
            except Exception:
                print "Trajectory.run_analysis() failed"
        else:
            raise Exception

    def run_analysis(self):  # FIXME: this seems obsolete
        self.test_if_agent()
        self.calc_kinematic_vals()  # evaluates kinematics


    def calc_kinematic_vals(self):
        ensemble = self.data
        # TODO: make wrapper function that iterates through trajectories in order to solve kinematics individually
        ensemble['curvature'] = calculate_curvature(ensemble)
        # absolute magnitude of velocity, accel vectors in 3D
        ensemble['velocity_magn'] = np.linalg.norm(ensemble.loc[:,('velocity_x', 'velocity_y', 'velocity_z')], axis=1)
        ensemble['acceleration_magn'] = np.linalg.norm(ensemble.loc[:,('acceleration_x', 'acceleration_y', 'acceleration_z')], axis=1)
        self._calc_polar_kinematics()


    def _calc_polar_kinematics(self):
        """append polar kinematics to vectors dictionary TODO: export to trajectory class"""
        kinematics = ['velocity', 'acceleration']
        if self.agent_obj is not None:  # we don't want to calculate the polar versions of these for experiments
            kinematics += ['randomF', 'wallRepulsiveF', 'upwindF', 'stimF']

        for name in kinematics:
            x_component, y_component = self.data[name+'_x'], self.data[name+'_y']
            self.data[name+'_xy_theta'] = calculate_xy_heading_angle(x_component, y_component)
            self.data[name+'_xy_mag'] = calculate_xy_magnitude(x_component, y_component)



    def plot_position_heatmaps(self):
        trimmed_df = self._trim_df_endzones()
        plotting.plot_position_heatmaps(trimmed_df, self.agent_obj)
        
    
    def plot_kinematic_hists(self, ensemble='none', titleappend=''):
        if type(ensemble) is str:
            ensemble = self.data
            ensemble = ensemble.loc[(ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]
        plotting.plot_kinematic_histograms(ensemble, self.agent_obj, titleappend=titleappend)
        
        
    def plot_door_velocity_compass(self, region='door', kind='avg_mag_per_bin'):
        
        if region == 'door':
            """plot the area """
            ensemble = self.data.loc[((self.data['position_x']>0.25) & (self.data['position_x']<0.5)), ['velocity_x', 'velocity_y']]
        else:
            ensemble = self.data
            
        plotting.plot_velocity_compassplot(ensemble, self.agent_obj, kind)
    

    def plot_kinematic_compass(self, kind='avg_mag_per_bin', data=None, flags='', title_append=''):
        if data is None: # TODO: wtf?
            data = self.data
        # TODO: add heading angle plot
        for vector_name in self.agent_obj['forces']+self.agent_obj['kinematic_vals']:
            # from enemble, selec mag and theta
            df = eval("data[['"+vector_name+"_xy_mag', '"+vector_name+"_xy_theta']]")
            # rename col to play nice w plotting function
            df.rename(columns={vector_name+'_xy_mag': 'magnitude', vector_name+'_xy_theta': 'angle'}, inplace=True)
            
            title = "{flag} Avg. {name} vector magnitude in xy plane (n = {N}) {titeappend}".format(\
            N = self.agent_obj['total_trajectories'], name=vector_name,\
            flag=flags, titeappend=title_append)
            fname = "{flag} Avg mag distribution of {name}_xy compass _".format(\
            name=vector_name, flag=flags)
  
            
            plotting.plot_compass_histogram(vector_name, df, self.agent_obj, title=title, fname=fname)
#            magnitudes, thetas = getattr(self.data, name+).values, getattr(V, name+'_xy_theta').values
#            plotting_funcs3D.compass_histogram(force, magnitudes, thetas, self.agent_obj)

       
    def plot_plume_triggered_compass(self, kind='avg_mag_per_bin'):
        behaviors = ['searching', 'entering', 'staying', 'Left_plume, exit left',
            'Left_plume, exit right', "Right_plume, exit left", "Right_plume, exit right"]
        for behavior in behaviors:
            ensemble = self.data.loc[self.data.plume_experience == behavior]
            if ensemble.empty is True:
                print "no data for ", behavior
                pass
            else:
                self.plot_kinematic_compass(data=ensemble, flags='Plume Triggered {}'.format(behavior))

    def plot_heading_compass(self):
        pass # TODO
        
        
    def plot_single_3Dtrajectory(self, trajectory_i=None):
        plot_kwargs = {'title': "{type} trajectory #{N}".format(type=self.is_agent, N=trajectory_i)}

        # get data
        if trajectory_i is None:
            trajectory_i = self.data.trajectory_num.min()
        selected_trajectory = self.get_trajectory_i(trajectory_i)

        plotting.plot3D_trajectory(selected_trajectory, plot_kwargs)

        
    def plot_sliced_hists(self):
        """Plot histograms from 0.25 < x < 0.95, as well as that same space
        divided into 4 equal 0.15 m segments 
        """
        x_edges = np.linspace(.25,.95,5)
        
        print "full ensemble"
        full_ensemble = self.data.loc[(self.data['position_x'] > x_edges[0]) \
            & (self.data['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=full_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[4]))
        
        print "downwind half ensemble"
        downwind_ensemble = self.data.loc[(self.data['position_x'] > x_edges[0]) \
            & (self.data['position_x'] < x_edges[2])]
        self.plot_kinematic_hists(ensemble=downwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[2]))

        print "upwind half ensemble"
        upwind_ensemble = self.data.loc[(self.data['position_x'] > x_edges[2]) \
            & (self.data['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=upwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[4]))    
#        
#        print "first quarter ensemble"
#        ensemble1 = self.data.loc[(self.data['position_x'] > x_edges[0]) \
#            & (self.data['position_x'] <= x_edges[1])]
#        self.plot_kinematic_hists(ensemble=ensemble1, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[1]))
#        
#        print "second quarter ensemble"
#        ensemble2 = self.data.loc[(self.data['position_x'] > x_edges[1]) \
#            & (self.data['position_x'] <= x_edges[2])]
#        self.plot_kinematic_hists(ensemble=ensemble2, titleappend=' {} < x <= {}'.format(x_edges[1], x_edges[2]))
#        
#        print "third quarter ensemble"
#        ensemble3 = self.data.loc[(self.data['position_x'] > x_edges[2]) \
#            & (self.data['position_x'] <= x_edges[3])]
#        self.plot_kinematic_hists(data=ensemble3, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[3]))
#
#        print "fourth quarter kinematics"
#        ensemble4 = self.data.loc[(self.data['position_x'] > x_edges[3]) \
#            & (self.data['position_x'] <= x_edges[4])]
#        self.plot_kinematic_hists(data=ensemble4, titleappend=' {} < x <= {}'.format(x_edges[3], x_edges[4]))
#        
        print "full- + up- + down-wind"
        plotting.plot_kinematic_histograms(full_ensemble, self.agent_obj, titleappend = '', upw_ensemble = upwind_ensemble, downw_ensemble = downwind_ensemble)


    def plume_stats(self):
        """ 
        in plume == 1, out == 0. therefore sum/n is % in plume
        """
        in_out = self.data.inPlume.values
        print "Total timesteps: {size}. Total time in plume: {vecsum}. Ratio:"\
            "{ratio}".format(size=in_out.size, vecsum=in_out.sum(), ratio=in_out.sum()/in_out.size)
            
            
    def describe(
            self,
            plot_kwargs={
                'trajectories':False,
                'heatmap':True,
                'states':True,
                'singletrajectories':False,
                'force_violin':True}):
        print self.data.describe()
        if plot_kwargs['heatmap'] is True:
            self.plot_position_heatmaps()
        if plot_kwargs['singletrajectories'] is True:
            self.plot_single_trajectory()
        if plot_kwargs['states'] is True:
    #         plot histogram of pos, velo, accel distributions
            self.plot_kinematic_hists()
        if plot_kwargs['force_violin'] is True:
            self.plot_force_violin()
            
            
    def dump2csvs(self):
        for trajectory_i in self.data.trajectory_num.unique():
            temp_traj = self.data[self.data['trajectory_num'] == trajectory_i]
            temp_array = temp_traj[['position_x', 'position_y']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")

    
    def calc_score(self):
        total_score, scores = score.score(self)
        print "total_score ", total_score
        print "scores", scores

    def _extract_number_from_fname(self, token):
        extract_digits = lambda stng: "".join(char for char in stng if char in string.digits + ".")
        to_float = lambda x: float(x) if x.count(".") <= 1 else None

        try:
            return to_float(extract_digits(token))
        except ValueError:
            print token, extract_digits(token)

    def get_trajectory_i(self, index):
        return self.data.loc[self.data['trajectory_num'] == index]

    def _trim_df_endzones(self):
        return self.data.loc[(self.data['position_x'] > 0.05) & (self.data['position_x'] < 0.95)]

    def test_if_agent(self):
        """Voight-Kampff test to test whether real mosquito or a robot
        Todo: is necessary?"""
        if self.agent_obj is None:
            type = "Mosquito"
        else:
            type = "Agent"

        self.is_agent = type


class Agent_Trajectory(Trajectory):
    def __init__(self):
        self.is_agent = "Agent"

    def visualize_forces(self):
        """like plot_vector_cloud, but with all of the kinematics at once"""
        plotting.plot_all_force_clouds(self.data)

    def plot_vector_cloud(self, kinematic='acceleration'):
        plotting.plot_vector_cloud(self.data, kinematic)

    def plot_force_violin(self):
        if self.agent_obj is None:
            raise TypeError('can\'t plot force violin for experimental data')
        plotting.plot_forces_violinplots(self.data, self.agent_obj)

    def add_agent_info(self, agent_obj):
        self.agent_obj = agent_obj


class Experimental_Trajectory(Trajectory):
    def __init__(self):
        self.is_agent = "Mosquito"
        self.agent_obj = None

    def load_experiments(self, selection=None):
        directory = i_o.get_directory(selection=selection)

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
            print "Loading " + fname
            file_path = os.path.join(directory, fname)
            base_name = os.path.splitext(file_path)[0]

            dataframe = pd.read_csv(file_path, na_values="NaN", names=col_labels)  # recognize str(NaN) as NaN
            dataframe.fillna(value=0, inplace=True)
            # take fname number, skip "control_" and ".csv"
            dataframe['trajectory_num'] = [int(fname[8:-4])] * dataframe.position_x.size
            df_list.append(dataframe)

        self.load_ensemble_and_analyze(data=df_list)
