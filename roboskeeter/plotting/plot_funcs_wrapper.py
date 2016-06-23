from roboskeeter.plotting import plot_kinematics, animate_trajectory_callable, plot_environment
import numpy as np


class PlotFuncsWrapper:
    def __init__(self, experiment, trim_endzones=False):
        """

        Parameters
        ----------
        experiment
            experiment class

        Returns
        -------
        None
        """
        self.experiment = experiment

        trajectory_type_dict = {True: 'Simulation', False: 'Experiment'}
        self.trajectory_type = trajectory_type_dict[self.experiment.is_simulation]

        self.observations = self.experiment.observations
        if trim_endzones:
            self.observations = self.observations.kinematics.loc[(self.observations.kinematics['position_x'] > 0.25) & \
                                                                 (self.observations.kinematics['position_x'] < 0.95)]

    def visualize_forces(self):
        """like plot_vector_cloud, but with all of the kinematics at once"""
        plot_kinematics.plot_all_force_clouds(self)

    def plot_force_violin(self):
        plot_kinematics.plot_forces_violinplots(self)

    def plot_vector_cloud(self, kinematic='acceleration', i=None):
        plot_kinematics.plot_vector_cloud(self, kinematic, i)

    def plot_vector_cloud_heatmap(self, kinematic='acceleration', i=None):
        plot_kinematics.vector_cloud_heatmap(self, kinematic, i)

    def plot_vector_cloud_kde(self, kinematic='acceleration', i=None):
        plot_kinematics.vector_cloud_kde(self, kinematic, i=None)

    def plot_vector_cloud_pairgrid(self, kinematic='acceleration', i=None):
        plot_kinematics.vector_cloud_pairgrid(self, kinematic, i=None)

    def plot_start_postiions(self):
        plot_kinematics.plot_starting_points(self.observations)

    def animate_trajectory(self, trajectory_i=None, highlight_inside_plume=False, show_plume=False, save =False, view='top'):
        if trajectory_i is None:
            trajectory_i = self.observations.get_trajectory_numbers().min()

        # get data
        selected_trajectory_df = self.observations.get_trajectory_slice(trajectory_i)

        p = selected_trajectory_df[['position_x', 'position_y', 'position_z']].values
        x_t = p.reshape((1, len(p), 3))  # make into correct shape for Jake vdp's code: 1 x T x 3

        fig, ax = plot_environment.plot_windtunnel(self.experiment.environment.windtunnel)
        ax.axis('off')

        if show_plume:
            plot_environment.draw_bool_plume(self.experiment.plume, ax=ax)

        anim = animate_trajectory_callable.Windtunnel_animation(fig, ax, x_t, save, view)
        anim.start_animation()

    def plot_position_heatmaps(self):
        plot_kinematics.plot_position_heatmaps(self)

    def plot_kinematic_hists(self):

        plot_kinematics.plot_kinematic_histograms(self.experiment)

    def plot_kinematic_compass(self, kind='avg_mag_per_bin', flags='', title_append=''):
        # TODO: add heading angle plot
        for vector_name in self.experiment.agent['forces'] + self.experiment.agent['kinematic_vals']:
            # from enemble, selec mag and theta
            df = eval("observations[['" + vector_name + "_xy_mag', '" + vector_name + "_xy_theta']]")
            # rename col to play nice w plotting function
            df.rename(columns={vector_name + '_xy_mag': 'magnitude', vector_name + '_xy_theta': 'angle'}, inplace=True)

            title = "{flag} Avg. {name} vector magnitude in xy plane (n = {N}) {titeappend}".format( \
                N=self.experiment.agent['total_trajectories'], name=vector_name, \
                flag=flags, titeappend=title_append)
            fname = "{flag} Avg mag distribution of {name}_xy compass _".format( \
                name=vector_name, flag=flags)

            plot_kinematics.plot_compass_histogram(vector_name, df, self.experiment.agent, title=title, fname=fname)
        #            magnitudes, thetas = getattr(self.data, name+).values, getattr(V, name+'_xy_theta').values
        #            plotting_funcs3D.compass_histogram(force, magnitudes, thetas, self.experiment.agent)

    def plot_plume_triggered_compass(self, kind='avg_mag_per_bin'):
        behaviors = ['searching', 'entering', 'staying', 'Left_plume, exit left',
                     'Left_plume, exit right', "Right_plume, exit left", "Right_plume, exit right"]
        for behavior in behaviors:
            ensemble = self.obvservations.kinematics.loc[self.obvservations.kinematics.plume_experience == behavior]
            if ensemble.empty is True:
                print "no data for ", behavior
                pass
            else:
                self.plot_kinematic_compass(data=ensemble, flags='Plume Triggered {}'.format(behavior))

    def plot_heading_compass(self):
        raise NotImplementedError  # TODO

    def plot_3Dtrajectory(self, trajectory_i=None, highlight_inside_plume=False, show_plume=False):
        if trajectory_i is None:
            trajectory_i = self.observations.get_trajectory_numbers().min()

        fig, ax = plot_environment.plot_windtunnel(self.experiment.environment.windtunnel)
        if show_plume:
            plot_environment.plot_windtunnel.draw_plume(self.experiment.plume, ax=ax)

        if trajectory_i is "ALL":
            index = self.observations.get_trajectory_numbers()
            ax.axis('off')
            for i in index:
                selected_trajectory_df = self.observations.get_trajectory_slice(i)
                plot_kwargs = {'title': "{type} trajectory #{N}".format(type=self.trajectory_type, N=i),
                               'highlight_inside_plume': highlight_inside_plume}
                plot_environment.draw_trajectory(ax, selected_trajectory_df)
        elif type(trajectory_i) is np.int64 or int:
            selected_trajectory_df = self.observations.get_trajectory_slice(trajectory_i)  # get data
            plot_kwargs = {'title': "{type} trajectory #{N}".format(type=self.trajectory_type, N=trajectory_i),
                           'highlight_inside_plume': highlight_inside_plume}
            plot_environment.draw_trajectory(ax, selected_trajectory_df, **plot_kwargs)
        else:
            raise TypeError("wrong kind of trajectory_index {}".format(type(trajectory_i)))

    def plot_sliced_hists(self):
        """Plot histograms from 0.25 < x < 0.95, as well as that same space
        divided into 4 equal 0.15 m segments
        """
        x_edges = np.linspace(.25, .95, 5)

        print "full ensemble"
        full_ensemble = self.obvservations.kinematics.loc[(self.obvservations.kinematics['position_x'] > x_edges[0]) \
                                            & (self.obvservations.kinematics['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=full_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[4]))

        print "downwind half ensemble"
        downwind_ensemble = self.obvservations.kinematics.loc[(self.obvservations.kinematics['position_x'] > x_edges[0]) \
                                                & (self.obvservations.kinematics['position_x'] < x_edges[2])]
        self.plot_kinematic_hists(ensemble=downwind_ensemble,
                                  titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[2]))

        print "upwind half ensemble"
        upwind_ensemble = self.obvservations.kinematics.loc[(self.obvservations.kinematics['position_x'] > x_edges[2]) \
                                              & (self.obvservations.kinematics['position_x'] < x_edges[4])]
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
        #        print "fourth quarter data"
        #        ensemble4 = self.data.loc[(self.data['position_x'] > x_edges[3]) \
        #            & (self.data['position_x'] <= x_edges[4])]
        #        self.plot_kinematic_hists(data=ensemble4, titleappend=' {} < x <= {}'.format(x_edges[3], x_edges[4]))
        #
        print "full- + up- + down-wind"
        plot_kinematics.plot_kinematic_histograms(full_ensemble, self.experiment.agent, titleappend='',
                                                  upw_ensemble=upwind_ensemble,
                                                  downw_ensemble=downwind_ensemble)

    def plot_score_comparison(self, reference_vals=None):
        plot_kinematics.plot_score_comparison(self.experiment, reference_vals=reference_vals)

    def plot_timeseries(self, kinematic=None):
        plot_kinematics.plot_timeseries(self.experiment, kinematic=kinematic)


