import numpy as np

from roboskeeter.math.math_toolbox import calculate_curvature, distance_from_wall


class DoMath:
    def __init__(self, experiment):
        self.experiment = experiment
        self.observations = experiment.observations

        self.calc_kinematic_vals()

        if self.experiment.experiment_conditions['optimizing'] is False:  # skip the following computations if optimizing
            self.percent_time_in_plume = self.calc_time_in_plume()
            self.side_ratio_score = self.calc_side_ratio_score()  # TODO: replace with KS test

            self.turn_threshold = 433.5
        else:
            self.percent_time_in_plume = self.side_ratio_score = self.turn_threshold = 0

    def calc_kinematic_vals(self):
        k = self.observations.kinematics

        k['curvature'] = calculate_curvature(k)
        # absolute magnitude of velocity, accel vectors in 3D
        k['velocity_norm'] = np.linalg.norm(k.loc[:, ('velocity_x', 'velocity_y', 'velocity_z')], axis=1)
        k['acceleration_norm'] = np.linalg.norm(k.loc[:, ('acceleration_x', 'acceleration_y', 'acceleration_z')],
                                                axis=1)
        k['dist_to_wall'] = distance_from_wall(k[['position_x', 'position_y', 'position_z']],
                                               self.experiment.environment.windtunnel.boundary)

    def calc_time_in_plume(self):
        """
        in plume == 1, out == 0. therefore sum/n is % in plume
        """
        in_out = self.observations.kinematics.in_plume.values
        N_timesteps = in_out.size
        try:
            N_timesteps_in_plume = in_out.sum()
            percent_time_in_plume = 1.0 * N_timesteps_in_plume / N_timesteps
        except ValueError:  # can't sum if not boolean plume  # FIXME: get working with timeavg plume
            percent_time_in_plume = "N/A-- not using boolean plume model"

        return percent_time_in_plume

    def calc_side_ratio_score(self):
        """upwind left vs right ratio"""  # TODO replace with KF score
        upwind_half = self.observations.kinematics.loc[self.observations.kinematics.position_x > 0.5]
        total_pts = len(upwind_half)

        left_upwind_pts = len(self.observations.kinematics.loc[(self.observations.kinematics.position_x > 0.5) & \
                                                               (self.observations.kinematics.position_y < 0)])
        right_upwind_pts = total_pts - left_upwind_pts
        # print "seconds extra on left side: ", (left_upwind_pts - right_upwind_pts) / 100.
        try:
            self.side_ratio_score = float(left_upwind_pts) / right_upwind_pts
        except ZeroDivisionError:
            print """\n Warning: no upwind right-side points found! This usually means you didn't run enough trajectories.
                  To avoid a divide 0 error we're skipping the side ratio scoring and setting side_ratio_score to 0."""
            self.side_ratio_score = 0

    def calc_KS(self):
        raise NotImplementedError  # TODO implement KS test




        #     self._calc_polar_kinematics()
        #
        #
        # def _calc_polar_kinematics(self):
        #     """append polar data to vectors dictionary"""
        #     data = ['velocity', 'acceleration']
        #     if self.agent_obj is not None:  # we don't want to calculate the polar versions of these for experiments
        #         data += ['randomF', 'wallRepulsiveF', 'upwindF', 'stimF']
        #
        #     for name in data:
        #         x_component, y_component = self.data[name+'_x'], self.data[name+'_y']
        #         self.data[name+'_xy_theta'] = calculate_xy_heading_angle(x_component, y_component)
        #         self.data[name+'_xy_mag'] = calculate_xy_magnitude(x_component, y_component)
        #
        #        self.data['turning'] = np.array([None] * self.max_bins)
        #     self.data['heading_angle'] = np.full(self.max_bins, np.nan)
        #     self.data['velocity_angular'] = np.full(self.max_bins, np.nan)

        #
        #
        # dist to wall vs velocity
        # ax = plt.hexbin(trajectory_s.data.dist_to_wall, trajectory_s.data.velocity_norm, cmap=plt.cm.RdBu, vmax=150) ; plt.colorbar(ax)
        # ax = plt.hexbin(trajectory_e.data.dist_to_wall, trajectory_e.data.velocity_norm, cmap=plt.cm.RdBu) ; plt.colorbar(ax)
