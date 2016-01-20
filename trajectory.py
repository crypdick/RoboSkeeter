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
import os
import string

import numpy as np
import pandas as pd

import scripts.plot_windtunnel as pwt
from score import Scoring
from scripts import i_o, animate_trajectory_callable
from scripts.math_sorcery import calculate_curvature, distance_from_wall


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


class Trajectory(object):
    def __init__(self, experiment):
        self.reload_data()
        self.agent_obj = None
        self.velo_kernel = None
        self.accel_kernel = None
        self.curvature_kernel = None
        self.is_agent = None
        self.experiment = experiment

    def reload_data(self):
        self.data = pd.DataFrame()

    def load_ensemble_and_analyze(self, data):
        if type(data) is dict:
            trajectory = pd.DataFrame(data)
            self.data.append(trajectory)  # this should be avoided b/c lots of overhead (slow)
            self.self.calc_kinematic_vals
        elif type(data) is list:
            self.data = pd.concat(data)  # fast
            self.calc_kinematic_vals()


    def calc_kinematic_vals(self):
        self.data['curvature'] = calculate_curvature(self.data)
        # absolute magnitude of velocity, accel vectors in 3D
        self.data['velocity_magn'] = np.linalg.norm(self.data.loc[:, ('velocity_x', 'velocity_y', 'velocity_z')],
                                                    axis=1)
        self.data['acceleration_magn'] = np.linalg.norm(
            self.data.loc[:, ('acceleration_x', 'acceleration_y', 'acceleration_z')], axis=1)
        self.data['dist_to_wall'] = distance_from_wall(self.data[['position_x', 'position_y', 'position_z']],
                                                       self.experiment.windtunnel.boundary)

    #     self._calc_polar_kinematics()
    #
    #
    # def _calc_polar_kinematics(self):
    #     """append polar kinematics to vectors dictionary"""
    #     kinematics = ['velocity', 'acceleration']
    #     if self.agent_obj is not None:  # we don't want to calculate the polar versions of these for experiments
    #         kinematics += ['randomF', 'wallRepulsiveF', 'upwindF', 'stimF']
    #
    #     for name in kinematics:
    #         x_component, y_component = self.data[name+'_x'], self.data[name+'_y']
    #         self.data[name+'_xy_theta'] = calculate_xy_heading_angle(x_component, y_component)
    #         self.data[name+'_xy_mag'] = calculate_xy_magnitude(x_component, y_component)
    #
    #        self.data['turning'] = np.array([None] * self.max_bins)
    #     self.data['heading_angle'] = np.full(self.max_bins, np.nan) TODO!
    #     self.data['velocity_angular'] = np.full(self.max_bins, np.nan)

    #
    #
    # dist to wall vs velocity
    # ax = plt.hexbin(trajectory_s.data.dist_to_wall, trajectory_s.data.velocity_magn, cmap=plt.cm.RdBu, vmax=150) ; plt.colorbar(ax)
    # ax = plt.hexbin(trajectory_e.data.dist_to_wall, trajectory_e.data.velocity_magn, cmap=plt.cm.RdBu) ; plt.colorbar(ax)

    def plot_position_heatmaps(self):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_position_heatmaps(self)
        
    
    def plot_kinematic_hists(self, ensemble='none', titleappend=''):
        import scripts.plotting_sorcery
        if type(ensemble) is str:
            ensemble = self.data
            ensemble = ensemble.loc[(ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]
        scripts.plotting_sorcery.plot_kinematic_histograms(ensemble, self.agent_obj, titleappend=titleappend)
        
        
    def plot_door_velocity_compass(self, region='door', kind='avg_mag_per_bin'):
        import scripts.plotting_sorcery
        
        if region == 'door':
            """plot the area """
            ensemble = self.data.loc[((self.data['position_x']>0.25) & (self.data['position_x']<0.5)), ['velocity_x', 'velocity_y']]
        else:
            ensemble = self.data

        scripts.plotting_sorcery.plot_velocity_compassplot(ensemble, self.agent_obj, kind)
    

    def plot_kinematic_compass(self, kind='avg_mag_per_bin', data=None, flags='', title_append=''):
        import scripts.plotting_sorcery
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

            scripts.plotting_sorcery.plot_compass_histogram(vector_name, df, self.agent_obj, title=title, fname=fname)
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
        raise NotImplementedError

    def plot_3Dtrajectory(self, trajectory_i=None, highlight_inside_plume=False, show_plume=False):
        if trajectory_i is None:
            trajectory_i = self.get_trajectory_numbers().min()

        fig, ax = pwt.plot_windtunnel(self.experiment.windtunnel)
        if show_plume:
            pwt.draw_plume(self.experiment.plume, ax=ax)

        if trajectory_i is "ALL":
            index = self.get_trajectory_numbers()
            ax.axis('off')
            for i in index:
                selected_trajectory_df = self.get_trajectory_i_df(i)
                plot_kwargs = {'title': "{type} trajectory #{N}".format(type=self.is_agent, N=i),
                               'highlight_inside_plume': highlight_inside_plume}
                pwt.draw_trajectory(ax, selected_trajectory_df)
        elif type(trajectory_i) is np.int64 or int:
            selected_trajectory_df = self.get_trajectory_i_df(trajectory_i)  # get data
            plot_kwargs = {'title': "{type} trajectory #{N}".format(type=self.is_agent, N=trajectory_i),
                           'highlight_inside_plume': highlight_inside_plume}
            pwt.draw_trajectory(ax, selected_trajectory_df, **plot_kwargs)
        else:
            raise TypeError("wrong kind of trajectory_i {}".format(type(trajectory_i)))




    def plot_sliced_hists(self):
        import scripts.plotting_sorcery
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
        scripts.plotting_sorcery.plot_kinematic_histograms(full_ensemble, self.agent_obj, titleappend='',
                                                           upw_ensemble=upwind_ensemble,
                                                           downw_ensemble=downwind_ensemble)

    def plot_score_comparison(self):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_score_comparison(self)

    def plot_timeseries(self, kinematic=None):
        """

        """
        # ensemble = self.data.loc[:,
        #            ['tsi', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z',
        #             'acceleration_x',
        #             'acceleration_y', 'acceleration_z', 'curvature']]
        #        ensemble['trajectory_num'] = ensemble.index
        # ensemble.reset_index(level=0, inplace=True)
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_timeseries(self.data, self.agent_obj, kinematic=kinematic)


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
        """we don't use self.data.to_csv(name) because Sharri likes having separate csvs for each trajectory
        """
        for trajectory_i in self.data.trajectory_num.unique():
            temp_traj = self.get_trajectory_i_df(trajectory_i)
            temp_array = temp_traj[['position_x', 'position_y']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")

    def calc_score(self, ref_ensemble='pickle'):
        scorer = Scoring()

        self.total_score = scorer.score_ensemble(self)
        print "Trajectory score: ", self.total_score

        return self.total_score

    def calc_side_ratio_score(self):
        """upwind left vs right ratio"""
        upwind_half = self.data.loc[self.data.position_x > 0.5]
        total_pts = len(upwind_half)

        left_upwind_pts = len(self.data.loc[(self.data.position_x > 0.5) & (self.data.position_y < 0)])
        right_upwind_pts = total_pts - left_upwind_pts
        print "seconds extra on left side: ", (left_upwind_pts - right_upwind_pts)/ 100.
        ratio = float(left_upwind_pts)/right_upwind_pts
        print ratio

    def _extract_number_from_fname(self, token):
        extract_digits = lambda stng: "".join(char for char in stng if char in string.digits + ".")
        to_float = lambda x: float(x) if x.count(".") <= 1 else None

        try:
            return to_float(extract_digits(token))
        except ValueError:
            print token, extract_digits(token)

    def get_trajectory_i_df(self, index):
        return self.data.loc[self.data.trajectory_num == int(index)]


    def _trim_df_endzones(self):
        return self.data.loc[(self.data['position_x'] > 0.05) & (self.data['position_x'] < 0.95)]

    #    def test_if_agent(self):
    #        """Voight-Kampff test to test whether real mosquito or a robot
    #        Todo: is necessary?"""
    #        if self.agent_obj is None:
    #            type = "Mosquito"
    #        else:
    #            type = "Agent"
    #
    #        self.is_agent = type

    #     def mk_iterator_from_col(self, col):
    # a.xs(0, level='trajectory_num')

    def get_trajectory_numbers(self):
        return np.sort(self.data.trajectory_num.unique())

    def plot_vector_cloud(self, kinematic='acceleration', i=None):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_vector_cloud(self, kinematic, i)

    def plot_vector_cloud_heatmap(self, kinematic='acceleration', i=None):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.vector_cloud_heatmap(self, kinematic, i)

    def plot_vector_cloud_kde(self, kinematic='acceleration', i=None):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.vector_cloud_kde(self, kinematic, i=None)

    def plot_vector_cloud_pairgrid(self, kinematic='acceleration', i=None):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.vector_cloud_pairgrid(self, kinematic, i=None)

    def plot_start_postiions(self):
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_starting_points(self.data)

    def animate_trajectory(self, trajectory_i=None, highlight_inside_plume=False, show_plume=False):
        if trajectory_i is None:
            trajectory_i = self.get_trajectory_numbers().min()

        # get data
        selected_trajectory_df = self.get_trajectory_i_df(trajectory_i)

        p = selected_trajectory_df[['position_x', 'position_y', 'position_z']].values
        x_t = p.reshape((1, len(p), 3))  # make into correct shape for Jake vdp's code: 1 x T x 3

        fig, ax = pwt.plot_windtunnel(self.experiment.windtunnel)
        ax.axis('off')

        if show_plume:
            pwt.draw_plume(self.experiment.plume, ax=ax)

        anim = animate_trajectory_callable.Windtunnel_animation(fig, ax, x_t)
        anim.start_animation()




class Agent_Trajectory(Trajectory):
    def __init__(self, experiment):
        super(Agent_Trajectory, self).__init__(experiment)
        # self.is_agent = "Agent"
        # self.experiment = experiment

    def visualize_forces(self):
        """like plot_vector_cloud, but with all of the kinematics at once"""
        import scripts.plotting_sorcery
        scripts.plotting_sorcery.plot_all_force_clouds(self.data)

    def plot_force_violin(self):
        import scripts.plotting_sorcery
        if self.agent_obj is None:
            raise TypeError('can\'t plot force violin for experimental data')
        scripts.plotting_sorcery.plot_forces_violinplots(self.data, self.agent_obj)

    def add_agent_info(self, agent_obj):
        self.agent_obj = agent_obj


class Experimental_Trajectory(Trajectory):
    def __init__(self, experiment):
        super(Experimental_Trajectory, self).__init__(experiment)
        # self.is_agent = "Mosquito"
        # self.agent_obj = None


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

        self.load_ensemble_and_analyze(data=df_list)