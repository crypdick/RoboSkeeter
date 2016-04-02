__author__ = 'richard'

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree as kdt
from analysis.plot_environment import plot_windtunnel as pwt
from scripts.i_o import get_directory


class Environment(object):
    def __init__(self, experiment):
        """Generate environmental objects

            experiment_kwargs = {'condition': 'Control',  # {'Left', 'Right', 'Control'}
                         'plume_model': "None" #"Boolean" "None, "Timeavg",
                         'time_max': "N/A (experiment)",
                         'bounded': True,
                         }
        """
        self.experiment_conditions = experiment.experiment_conditions

        for key in self.experiment_conditions:
            setattr(self, key, self.experiment_conditions[key])

        # Load windtunnel
        self.windtunnel = Windtunnel(self.condition)

        # Load correct plume
        if self.plume_model == "Boolean":
            self.plume = Boolean_Plume(self)
        elif self.plume_model == "timeavg":
            self.plume = Timeavg_Plume(self)
        elif self.plume_model == "None":
            self.plume = Plume(self)
        else:
            raise NotImplementedError("no such plume type {}".format(self.plume_model))


class Windtunnel:
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
        ax = pwt.plot_windtunnel(self)
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

    def in_bounds(self, position):
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
        ''' given {left, right, none, custom coords} place heater in the windtunnel

        Args:
        location

        returns [x,y, zmin, zmax, diam]
        '''
        self.side = side
        self.experimental_condition = experimental_condition

        if side == experimental_condition:
            self.is_on = True
        else:
            self.is_on = False

        colors = {False: 'black', True: 'red'}
        self.color = colors[self.is_on]

        self.zmin = 0.03800
        self.zmax = 0.11340
        self.diam = 0.00635
        (self.x_position, self.y_position) = self._set_xy_coords()

    def _set_xy_coords(self):
        x_coord = 0.864

        if self.side in "leftLeftLEFT":
            y_coord = -0.0507
        elif self.side in "rightRightRIGHT":
            y_coord = 0.0507
        elif self.side in 'controlControlCONTROL':
            x_coord, y_coord = None, None
        else:
            raise Exception('invalid location type specified')

        return (x_coord, y_coord)


class Plume(object):
    ''' The Plume superclass

    Args:
    condition: {left|right|None}
    '''

    def __init__(self, experiment):
        # useful aliases
        self.experiment = experiment
        self.condition = experiment.condition
        self.walls = experiment.windtunnel.walls


class Boolean_Plume(Plume):
    """Are you in the plume Y/N"""
    def __init__(self, experiment):
        super(self.__class__, self).__init__(experiment)

        self.data = self._load_plume_data()

        try:
            self.resolution = abs(self.data.x_position.diff()[1])
        except AttributeError:  # if no plume, can't take diff() of no data
            self.resolution = None

    def in_plume(self, position):
        in_bounds, _ = self.walls.in_bounds(position)
        x, y, z = position

        if self.condition in 'controlControlCONTROL':
            inPlume = False
        elif np.abs(self.data['x_position'] - x).min() > self.resolution:
            # too far from the plume in the upwind/downwind direction
            inPlume = False
        elif in_bounds is False:
            print("WARNING: sniffing outside of windtunnel bounds")
            inPlume = False
        else:
            plume_plane = self._get_nearest_plume_plane(x)
            minor_axis = plume_plane.small_radius
            minor_ax_major_ax_ratio = 3
            major_axis = minor_axis * minor_ax_major_ax_ratio

            # implementation of http://math.stackexchange.com/a/76463/291217

            value = (((y - plume_plane.y_position) ** 2) / minor_axis ** 2) + \
                    (((z - plume_plane.z_position) ** 2) / major_axis ** 2)

            if value <= 1:
                inPlume = True
            else:
                inPlume = False

        return inPlume

    def show(self):
        fig, ax = pwt.plot_windtunnel(self.experiment.windtunnel)
        ax.axis('off')
        pwt.draw_plume(self, ax=ax)

    def _get_nearest_plume_plane(self, x_position):
        """given x position, find nearest plan"""
        closest_plume_index = (np.abs(self.data['x_position'] - x_position)).argmin()
        plume_plane = self.data.loc[closest_plume_index]

        return plume_plane

    def _load_plume_data(self):
        col_names = ['x_position', 'z_position', 'small_radius']

        if self.condition in 'controlControlCONTROL':
            return None
        elif self.condition in 'lLleftLeft':
            plume_dir = get_directory('BOOL_LEFT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.experiment.windtunnel.heater_l.y_position
        elif self.condition in 'rightRight':
            plume_dir = get_directory('BOOL_RIGHT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.experiment.windtunnel.heater_r.y_position
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        return df


class Timeavg_Plume(Plume):
    """time-averaged temperature readings taken inside the windtunnel"""
    def __init__(self, experiment):
        super(self.__class__, self).__init__(experiment)

        # useful references
        self.left = -0.127
        self.right = 0.127
        self.upwind = 1.0
        self.downwind = 0.0
        self.ceiling = 0.254
        self.floor = 0.
        self.boundary = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]

        # initialize vals
        self.data = pd.DataFrame()
        self._grid_x, self._grid_y, self.grid_z, self._interpolated_temps = None, None, None, None
        self._gradient_x, self._gradient_y, self._gradient_z = None, None, None

        # number of x, y, z positions to interpolate the data. numbers chosen to reflect the spacing at which the measurements
        # were taken to avoid gradient values of 0 due to undersampling
        resolution = (100j, 25j, 25j) # stored as complex numbers for mgrid to work properly

        self._raw_data = self._load_plume_data()
        self._interpolate_data(resolution)
        self._calc_gradient()
        self.tree = self._calc_kdtree()
        print """Warning: we don't know the plume bounds for the Timeavg plume, so the in_plume() method
                always returns False"""

    def in_plume(self, position):
        """we don't know the plume bounds for the Timeavg plume, so we're returning False always as a dummy value """
        return False

    def show_gradient(self, thresh = 0):
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        filt = self.data[self.data.gradient_mag > thresh]

        ax.quiver(filt.x, filt.y, filt.z, filt.gradient_x, filt.gradient_y, filt.gradient_z, length=0.01)
        # ax.set_xlim3d(0, 1)
        ax.set_ylim3d(-0.127, 0.127)
        ax.set_zlim3d(0, 0.254)

        plt.title("Temperature gradient of interpolated time-averaged thermocouple recordings")
        plt.xlabel("Upwind/downwind")
        plt.ylabel("Crosswind")
        plt.clabel("Elevation")

        plt.show()

    def get_nearest(self, location):
        """given [x,y,z] return nearest

        query() returns """

        _, index = self.tree.query(location)
        return self.data.iloc[index]

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
        xmin = self._raw_data.x.min()
        xmax = self._raw_data.x.max()
        ymin = self._raw_data.y.min()
        ymax = self._raw_data.y.max()
        zmin = self._raw_data.z.min()
        zmax = self._raw_data.z.max()

        self.downwind

    def _interpolate_data(self, resolution):
        if self.condition in 'controlControlCONTROL':
            return None

        # self._grid_x, self._grid_y, self._grid_z = np.mgrid[0.:1.:resolution[0], -0.127:0.127:resolution[1], 0:0.254:resolution[2]]
        # grid_x, grid_y, grid_z = np.mgrid[0.:1.:100j, -0.127:0.127:25j, 0:0.254:25j]
        print len(np.unique(self._raw_data.x)),len(np.unique(self._raw_data.y)), len(np.unique(self._raw_data.z))
        # self._raw_data['x'] = self._raw_data.x / 5.
        # self._grid_x = self._grid_x / 5.
        # points = self._raw_data[['x', 'y', 'z']].values  # (1382, 3)
        # points = (self._raw_data.x.values, self._raw_data.y.values, self._raw_data.z.values)  # (1382, 3)
        x,y,z = (self._raw_data.x.values, self._raw_data.y.values, self._raw_data.z.values)  # (1382, 3)
        print "pts", len(self._raw_data.x.values), np.shape(self._raw_data.x.values)
        temps = self._raw_data.temperature.values  # len 1382
        print "temps", temps.shape
        epsilon = 3
        print "epsilon", epsilon
        rbfi = Rbf(x,y,z, temps, function='gaussian', smooth=1e-8, epsilon=epsilon)

        xi = np.linspace(0, 1, 50)  # xmin * .8
        yi = np.linspace(-.127, .127, 15)
        zi = np.linspace(0, .254, 15)
        print "array shapes", xi.shape, yi.shape, zi.shape
        self._grid_x, self._grid_y, self._grid_z = np.meshgrid(xi,yi,zi, indexing='ij')
        xxi = self._grid_x.ravel()  # FIXME flip here
        yyi = self._grid_y.ravel()
        zzi = self._grid_z.ravel()
        print "xxi shape", np.shape(xxi), np.shape(yyi), np.shape(zzi)
        print "grid shapes", np.shape(self._grid_x)
        di = rbfi(xxi,yyi,zzi)
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

        # FIXME: set out of bounds temp to room temp


    def _calc_gradient(self):
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




if __name__ == '__main__':
    windtunnel = Windtunnel('left')