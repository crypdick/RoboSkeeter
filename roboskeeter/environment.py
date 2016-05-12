__author__ = 'richard'

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree as kdt

from roboskeeter.io.i_o import get_directory
from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_gradient, draw_bool_plume


class Environment(object):
    def __init__(self, experiment):
        """
        Generate environmental objects

        Parameters
        ----------
        experiment
            (object)
        """
        self.condition = experiment.experiment_conditions['condition']
        self.bounded = experiment.experiment_conditions['bounded']
        self.plume_model = experiment.experiment_conditions['plume_model'].lower()

        if self.condition == 'Control' and self.plume_model != 'none':
            print "{} plume model selected for control condition, setting instead to no plume.".format(self.plume_model)
            self.plume_model = 'none'

        self.windtunnel = WindTunnel(self.condition)
        self.plume = self._load_plume()
        self.room_temperature = 19.0

    def _load_plume(self):
        if self.plume_model == "boolean":
            plume = BooleanPlume(self)
        elif self.plume_model == "timeavg":
            plume = TimeAvgPlume(self)
        elif self.plume_model == "none":
            plume = NoPlume(self)
        elif self.plume_model == "unaveraged":
            plume = UnaveragedPlume(self)
        else:
            raise NotImplementedError("no such plume type {}".format(self.plume_model))

        return plume


class WindTunnel:
    def __init__(self, experimental_condition):
        """
        experimental_condition

        """
        self.walls = Walls()
        self.boundary = self.walls.boundary
        self.experimental_condition = experimental_condition

        self.heater_l = Heater("Left", self.experimental_condition)
        self.heater_r = Heater("Right", self.experimental_condition)

    def show(self):
        fig, ax = plot_windtunnel(self)
        return fig, ax


class Walls:
    def __init__(self):
        # these are real dims of our wind tunnel
        self.left = -0.127
        self.right = 0.127
        self.upwind = 1.0
        self.downwind = 0.0
        self.ceiling = 0.254
        self.floor = 0.
        self.boundary = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]

    def check_in_bounds(self, position):
        xpos, ypos, zpos = position
        inside = True
        past_wall = []

        if xpos > self.upwind:  # beyond upwind(upwind) wall (end)
            inside = False
            past_wall.append('upwind')
        if xpos < self.downwind:  # too far behind
            inside = False
            past_wall.append('downwind')
        if ypos < self.left:  # too left
            inside = False
            past_wall.append('left')
        if ypos > self.right:  # too far right
            inside = False
            past_wall.append('right')
        if zpos > self.ceiling:  # too far above
            inside = False
            past_wall.append('ceiling')
        if zpos < self.floor:  # too far below
            inside = False
            past_wall.append('floor')

        return inside, past_wall


class Heater:
    def __init__(self, side, experimental_condition):
        """
        given side, generate heater

        Parameters
        ----------
        side
            {left, right, none, custom coords}
            Location of the heater
        experimental_condition
            determines whether the heater is on or off
        Returns
        -------
        None
        """
        self.side = side
        self.experimental_condition = experimental_condition

        if side == experimental_condition:
            self.is_on = True
            self.color = 'red'
        else:
            self.is_on = False
            self.color = 'black'

        self.zmin, self.zmax, self.diam, self.x_position, self.y_position = self._set_coordinates()

    def _set_coordinates(self):
        x_coord = 0.864
        zmin = 0.03800
        zmax = 0.11340
        diam = 0.00635

        if self.side in "leftLeftLEFT":
            y_coord = -0.0507
        elif self.side in "rightRightRIGHT":
            y_coord = 0.0507
        elif self.side in 'controlControlCONTROL':
            x_coord, y_coord = None, None
        else:
            raise Exception('invalid location type specified')

        return zmin, zmax, diam, x_coord, y_coord


class Plume(object):
    def __init__(self, environment):
        """
        The plume base class
        Parameters
        ----------
        environment
            (object)

        Returns
        -------
        """
        # useful aliases
        self.environment = environment
        self.condition = environment.condition
        self.walls = environment.windtunnel.walls
        self.plume_model = environment.plume_model


class NoPlume(Plume):
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

    def check_for_plume(self, _):
        # always return false
        return False


class BooleanPlume(Plume):
    """Are you in the plume Y/N"""
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

        self.data = self._load_plume_data()

        self.resolution = self._calc_resolution()

    def check_for_plume(self, position):
        in_bounds, _ = self.walls.check_in_bounds(position)
        x, y, z = position

        if np.abs(self.data['x_position'] - x).min() > self.resolution:
            # calcs distance to all points, if too far from the plume in the upwind/downwind direction returns false
            in_plume = False
        elif in_bounds is False:
            print("WARNING: can't find plumes outside of windtunnel bounds")
            in_plume = False
        else:
            plume_plane = self._get_nearest_plume_plane(x)
            minor_axis = plume_plane.small_radius
            minor_ax_major_ax_ratio = 3
            major_axis = minor_axis * minor_ax_major_ax_ratio

            # check if position is within the elipsoid
            # implementation of http://math.stackexchange.com/a/76463/291217
            value = (((y - plume_plane.y_position) ** 2) / minor_axis ** 2) + \
                    (((z - plume_plane.z_position) ** 2) / major_axis ** 2)

            if value <= 1:
                in_plume = True
            else:
                in_plume = False

        return in_plume

    def show(self):
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        ax.axis('off')
        draw_bool_plume(self, ax=ax)

    def _get_nearest_plume_plane(self, x_position):
        """given x position, find nearest plan"""
        closest_plume_index = (np.abs(self.data['x_position'] - x_position)).argmin()
        plume_plane = self.data.loc[closest_plume_index]

        return plume_plane

    def _load_plume_data(self):
        col_names = ['x_position', 'z_position', 'small_radius']

        if self.condition in 'controlControlCONTROL':
            raise Exception("This block shouldn't ever run.")
        elif self.condition in 'lLleftLeft':
            plume_dir = get_directory('BOOL_LEFT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.environment.windtunnel.heater_l.y_position
        elif self.condition in 'rightRight':
            plume_dir = get_directory('BOOL_RIGHT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.environment.windtunnel.heater_r.y_position
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        return df

    def _calc_resolution(self):
        """ use x data to calculate the resolution"""
        try:
            resolution = abs(self.data.x_position.diff()[1])
        except AttributeError:  # if no plume, can't take diff() of no data
            resolution = None

        return resolution


class TimeAvgPlume(Plume):
    """time-averaged temperature readings taken inside the windtunnel"""
    # TODO: test TimeAvgPlume
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

        # useful references
        self.left = self.walls.left
        self.right = self.walls.right
        self.upwind = self.walls.upwind
        self.downwind = self.walls.downwind
        self.ceiling = self.walls.ceiling
        self.floor = self.walls.floor
        self.bounds = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]

        # number of x, y, z positions to interpolate the data. numbers chosen to reflect the spacing at which the
        # measurements were taken to avoid gradient values of 0 due to undersampling
        # resolution = (100j, 25j, 25j)  # stored as complex numbers for mgrid to work properly
        interpolation_resolution = .05  # 1 cm

        self._raw_data = self._load_plume_data()
        self.padded_data = self._pad_plume_data()
        self.data, self.grid_x, self.grid_y, self.grid_z, self.grid_temp = self._interpolate_data(self.padded_data, interpolation_resolution)
        self.gradient_x, self.gradient_y, self.gradient_z = self._calc_gradient()
        self.tree = self._calc_kdtree()

        print """Timeaveraged plume stats:  TODO implement sanity checks
        raw data min temp: {}
        raw data max temp: {}
        interpolated min temp: {}
        interpolated max temp: {}
        """.format(self._raw_data.avg_temp.min(),self._raw_data.avg_temp.max(),
                   self.data.avg_temp.min(), self.data.avg_temp.max())

        print """Warning: we don't know the plume bounds for the Timeavg plume, so the check_for_plume() method
                always returns False"""

    def check_for_plume(self, _):
        """
        we don't know the plume bounds for the Timeavg plume, so the check_for_plume() method
                always returns False

        Returns
        -------
        in_plume
            always return False.
        """

        return False

    def get_nearest_prediction(self, position):
        """
        Given [x,y,z] return nearest temperature data
        Parameters
        ----------
        position
            [x,y,z]

        Returns
        -------
        temperature
        """

        _, index = self.tree.query(position)
        data = self.data.iloc[index]
        return data['avg_temp']

    def get_nearest_gradient(self, position):
        data = self.get_nearest_prediction(position)
        return np.array([data['gradient_x'], data['gradient_y'], data['gradient_z']])

    def show_raw_data(self):
        from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_recordings_scatter
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        plot_plume_recordings_scatter(self._raw_data, ax)
        fig.show()

    def show(self):
        import roboskeeter.plotting.plot_environment_mayavi as pemavi
        # fig, ax = plot_windtunnel(self.environment.windtunnel)
        # plot_plume_recordings_scatter(self.data, ax)
        pemavi.plot_plume_recordings_volume(self.bounds, self.grid_x, self.grid_y, self.grid_z, self.grid_temp)
        # fig.show()


    def plot_gradient(self, thresh=0):
        from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_gradient
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        plot_plume_gradient(self._gradient_x, ax, thresh)  # FIXME use all gradients
        fig.show()

    def _load_plume_data(self):
        col_names = ['x', 'y', 'z', 'avg_temp']

        if self.condition in 'controlControlCONTROL':
            return None
        elif self.condition in 'lLleftLeft':
            plume_dir = get_directory('THERMOCOUPLE_TIMEAVG_LEFT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
        elif self.condition in 'rightRight':
            plume_dir = get_directory('THERMOCOUPLE_TIMEAVG_RIGHT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        return df.dropna()

    def _pad_plume_data(self):
        """
        We are assuming that far away from the plume envelope the air will be room temperature. We are padding the
        recorded area with room temperature data points

        Appends the padded data to the raw data
        """
        # expand the windtunnel bounds to synthesize data outside the windtunnel bounds
        expand_factor = .4  # expand by this much
        y_pad = (self.right - self.left) * expand_factor
        x_pad = (self.upwind - self.downwind) * expand_factor
        z_pad = (self.ceiling - self.floor) * expand_factor

        left = self.left - y_pad
        right = self.right + y_pad
        floor =  self.floor - z_pad
        ceiling = self.ceiling + z_pad
        downwind = self.downwind - x_pad
        upwind = self.upwind + x_pad

        padding_distance = 0.03  # start padding 3 cm away from recorded data

        data_xmin = self._raw_data.x.min() - padding_distance
        data_xmax = self._raw_data.x.max() + padding_distance
        data_ymin = self._raw_data.y.min() - padding_distance
        data_ymax = self._raw_data.y.max() + padding_distance
        data_zmin = self._raw_data.z.min() - padding_distance
        data_zmax = self._raw_data.z.max() + padding_distance


        df_list = [self._raw_data]  # append to raw data
        # make grids of room temp data to fill the volume surrounding the place we took measurements
        df_list.append(self._make_uniform_data_grid(downwind, data_xmin, left, right, floor, ceiling))
        df_list.append(self._make_uniform_data_grid(data_xmax, upwind, left, right, floor, ceiling))

        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, left, data_ymin, floor, ceiling))
        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymax, right, floor, ceiling))

        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymin, data_ymax, floor, data_zmin))
        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymin, data_ymax, data_zmax, ceiling))

        return pd.concat(df_list)

    def _make_uniform_data_grid(self, xmin, xmax, ymin, ymax, zmin, zmax, temp=19, res=0.1):
        # left grid
        x = np.arange(xmin, xmax, res)
        y = np.arange(ymin, ymax, res)
        z = np.arange(zmin, zmax, res)
        # we save this grid b/c it helps us with the gradient func
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        df_dict = dict()
        df_dict['x'] = xx.ravel()  # FIXME flip here
        df_dict['y'] = yy.ravel()
        df_dict['z'] = zz.ravel()
        df_dict['avg_temp'] = np.array([temp] * len(x) * len(y) * len(z))

        df = pd.DataFrame(data=df_dict)

        return df

    def _interpolate_data(self, data, resolution):
        """
        Replace data with a higher resolution interpolation
        Parameters
        ----------
        data
        resolution

        Returns
        -------
        interpolated_temps, (grid_x, grid_y, grid_z, grid_temps)
        """
        # TODO: review this function
        if self.condition in 'controlControlCONTROL':
            return None  # TODO: wtf

        # useful aliases
        x, y, z, temps = data.x.values, data.y.values, data.z.values, data.avg_temp.values

        # init rbf interpolator
        epsilon = .06  # TODO: set as the average 3d euclidean distance between observations
        smoothing = .1
        rbfi = Rbf(x, y, z, temps, function='gaussian', smooth=smoothing, epsilon=epsilon)

        # make positions to interpolate at
        xi = np.linspace(self.downwind, self.upwind, 50)  # TODO fix resolution
        yi = np.linspace(self.left, self.right, 15)
        zi = np.linspace(self.floor, self.ceiling, 15)
        # we save this grid b/c it helps us with the gradient func
        grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')
        grid_x_flat = grid_x.ravel()  # FIXME flip here
        grid_y_flat = grid_y.ravel()
        grid_z_flat = grid_z.ravel()

        # interpolate
        interp_temps = rbfi(grid_x_flat, grid_y_flat, grid_z_flat)
        # we save this grid b/c it helps us with the gradient func
        grid_temps = interp_temps.reshape((len(xi), len(yi), len(zi)))

        # save to df
        df_dict = dict()
        df_dict['x'] = grid_x_flat
        df_dict['y'] = grid_y_flat
        df_dict['z'] = grid_z_flat
        df_dict['avg_temp'] = interp_temps
        interpolated_temps = pd.DataFrame(df_dict)

        return interpolated_temps, grid_x, grid_y, grid_z, grid_temps

    def _calc_gradient(self):
        # return None # TODO: awaiting fix to gradient function: https://stackoverflow.com/questions/36781698/numpy-sample-distances-for-3d-gradient
        # TODO: review this gradient function
        if self.condition in 'controlControlCONTROL':
            return None

        # Solve for the spatial
        distances = [np.diff(self.grid_x, axis=0)[0][0][0], np.diff(self.grid_y, axis=1)[0][0][0], np.diff(self.grid_z, axis=2)[0][0][0]]
        gradient_x, gradient_y, gradient_z = np.gradient(self.grid_temp, *distances)

        self.data['gradient_x'] = gradient_x.ravel()
        self.data['gradient_y'] = gradient_y.ravel()
        self.data['gradient_z'] = gradient_z.ravel()

        self.data.fillna(0, inplace=True)  # replace NaNs, infs before calculating norm
        self.data.replace([np.inf, -np.inf], 0, inplace=True)

        self.data['gradient_norm'] = np.linalg.norm(self.data[['gradient_x', 'gradient_y', 'gradient_z']], axis=1)

        return gradient_x, gradient_y, gradient_z

    def _calc_kdtree(self):
        if self.condition in 'controlControlCONTROL':
            return None

        data = zip(self.data.x, self.data.y, self.data.z)
        return kdt(data)


class UnaveragedPlume:
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)
        raise NotImplementedError  # TODO: implement unaveraged plume