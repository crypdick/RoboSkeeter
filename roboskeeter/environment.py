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
        self.plume_model = experiment.experiment_conditions['plume_model']

        if self.condition == 'Control' and self.plume_model != 'None':
            print "{} plume model selected for control condition, setting instead to no plume.".format(self.plume_model)
            self.plume_model = 'None'

        self.windtunnel = WindTunnel(self.condition)
        self.plume = self._load_plume()

    def _load_plume(self):
        if self.plume_model == "Boolean":
            plume = BooleanPlume(self)
        elif self.plume_model == "timeavg":
            plume = TimeAvgPlume(self)
        elif self.plume_model == "None":
            plume = NoPlume(self)
        elif self.plume_model == "Unaveraged":
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
        ax = plot_windtunnel(self)
        return ax


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

        # initialize vals
        self.data = pd.DataFrame()
        self._grid_x, self._grid_y, self.grid_z, self._interpolated_temps = None, None, None, None
        self._gradient_x, self._gradient_y, self._gradient_z = None, None, None

        # number of x, y, z positions to interpolate the data. numbers chosen to reflect the spacing at which the
        # measurements were taken to avoid gradient values of 0 due to undersampling
        resolution = (100j, 25j, 25j)  # stored as complex numbers for mgrid to work properly

        self._raw_data = self._load_plume_data()
        self._interpolate_data(resolution)
        self._calc_gradient()
        self.tree = self._calc_kdtree()
        print """Warning: we don't know the plume bounds for the Timeavg plume, so the in_plume() method
                always returns False"""

    def check_for_plume(self, _):
        """

        Returns
        -------
        in_plume
            always return False.
        """
        return False

    def plot_gradient(self, thresh=0):
        plot_plume_gradient(self, thresh)

    def get_nearest_data(self, position):
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
        return data

    def get_nearest_temp(self, position):
        data = self.get_nearest_data(position)
        temp = data['avg_temp']
        return temp

    def get_nearest_gradient(self, position):
        data = self.get_nearest_data(position)
        return np.array([data['gradient_x'], data['gradient_y'], data['gradient_z']])

    def _load_plume_data(self):
        col_names = ['x', 'y', 'z', 'temperature']

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
        """
        # TODO: pad the data extending outside of the windtunnel bounds.
        xmin = self._raw_data.x.min()
        xmax = self._raw_data.x.max()
        ymin = self._raw_data.y.min()
        ymax = self._raw_data.y.max()
        zmin = self._raw_data.z.min()
        zmax = self._raw_data.z.max()

        self.downwind # FIXME

        raise NotImplementedError # TODO; implement padding

    def _interpolate_data(self, resolution):
        # TODO: review this function
        if self.condition in 'controlControlCONTROL':
            return None

        # self._grid_x, self._grid_y, self._grid_z = np.mgrid[0.:1.:resolution[0], -0.127:0.127:resolution[1], 0:0.254:resolution[2]]
        # grid_x, grid_y, grid_z = np.mgrid[0.:1.:100j, -0.127:0.127:25j, 0:0.254:25j]
        #print len(np.unique(self._raw_data.x)),len(np.unique(self._raw_data.y)), len(np.unique(self._raw_data.z))
        # self._raw_data['x'] = self._raw_data.x / 5.
        # self._grid_x = self._grid_x / 5.
        # points = self._raw_data[['x', 'y', 'z']].values  # (1382, 3)
        # points = (self._raw_data.x.values, self._raw_data.y.values, self._raw_data.z.values)  # (1382, 3)
        x, y, z = (self._raw_data.x.values, self._raw_data.y.values, self._raw_data.z.values)  # (1382, 3)
        #print "pts", len(self._raw_data.x.values), np.shape(self._raw_data.x.values)
        temps = self._raw_data.temperature.values  # len 1382
        #print "temps", temps.shape
        epsilon = 3
        #print "epsilon", epsilon
        rbfi = Rbf(x, y, z, temps, function='gaussian', smooth=1e-8, epsilon=epsilon)

        xi = np.linspace(0, 1, 50)  # xmin * .8
        yi = np.linspace(-.127, .127, 15)
        zi = np.linspace(0, .254, 15)
        #print "array shapes", xi.shape, yi.shape, zi.shape
        self._grid_x, self._grid_y, self._grid_z = np.meshgrid(xi, yi, zi, indexing='ij')
        xxi = self._grid_x.ravel()  # FIXME flip here
        yyi = self._grid_y.ravel()
        zzi = self._grid_z.ravel()
        print "xxi shape", np.shape(xxi), np.shape(yyi), np.shape(zzi)
        print "grid shapes", np.shape(self._grid_x)
        di = rbfi(xxi, yyi, zzi)
        print "shape di", di.shape
        self._interpolated_temps = di.reshape((len(xi), len(yi), len(zi)))
        # print self._interpolated_temps


        # self._interpolated_temps = griddata(points,
        #                               temps,
        #                               (self._grid_x, self._grid_y, self._grid_z),
        #                               method='linear')
        #
        # self._interpolated_temps = interpn(points,
        #                               temps,
        #                               (self._grid_x, self._grid_y, self._grid_z),
        #                               method='linear')

        # self.data['x'] = self._grid_x.ravel()
        # self.data['y'] = self._grid_y.ravel()
        # self.data['z'] = self._grid_z.ravel()
        # self.data['avg_temp'] = self._interpolated_temps.ravel()
        self.data['x'] = xxi
        self.data['y'] = yyi
        self.data['z'] = zzi
        self.data['avg_temp'] = di

    def _calc_gradient(self):
        # TODO: review this gradient function
        if self.condition in 'controlControlCONTROL':
            return None

        # Solve for the spatial gradient
        self._gradient_x, self._gradient_y, self._gradient_z = np.gradient(self._interpolated_temps,
                                                                           self._grid_x,
                                                                           self._grid_y,
                                                                           self._grid_z)

        self.data['gradient_x'] = self._gradient_x.ravel()
        self.data['gradient_y'] = self._gradient_y.ravel()
        self.data['gradient_z'] = self._gradient_z.ravel()

        print """raw min {}
                raw max {}
                interp min {}
                interp max {}
                """.format(self._raw_data.temperature.min(),self._raw_data.temperature.max(),
                           self.data.avg_temp.min(),self.data.avg_temp.max())

        self.data.fillna(0, inplace=True)  # replace NaNs, infs before calculating norm
        self.data.replace([np.inf, -np.inf], 0, inplace=True)

        self.data['gradient_mag'] = np.linalg.norm(self.data[['gradient_x', 'gradient_y', 'gradient_z']], axis=1)

    def _calc_kdtree(self):
        if self.condition in 'controlControlCONTROL':
            return None

        data = zip(self.data.x, self.data.y, self.data.z)
        return kdt(data)


class UnaveragedPlume:
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)
        raise NotImplementedError  # TODO: implement unaveraged plume