__author__ = 'richard'

import numpy as np
import pandas as pd

import scripts.plot_windtunnel as pwt


class Windtunnel():
    def __init__(self, experimental_condition):
        """
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)
        collision type:
            'elastic', 'crash'

        TODO: class  Wall. left wall right wall

        """
        self.walls = Walls()
        self.boundary = self.walls.boundary
        self.experimental_condition = experimental_condition

        self.heater_l = Heater("Left", self.experimental_condition)
        self.heater_r = Heater("Right", self.experimental_condition)

    def show(self):
        ax = pwt.plot_windtunnel(self)
        return ax

class Walls():
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



class Heater():
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
    ''' Instantiates a plume object.

    Args:
    condition: {left|right|None}
    '''

    def __init__(self, experiment):
        # useful aliases
        self.experiment = experiment
        self.condition = experiment.condition
        self.walls = experiment.windtunnel.walls


class Boolean_Plume(Plume):
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

    def load_plume_data(self):
        col_names = ['x_position', 'z_position', 'small_radius']

        if self.condition in 'controlControlCONTROL':
            return None
        elif self.condition in 'lLleftLeft':
            df = pd.read_csv('data/experiments/plume_data/left_plume_bounds.csv', names=col_names)
            df['y_position'] = self.experiment.windtunnel.heater_l.y_position
        elif self.condition in 'rightRight':
            df = pd.read_csv('data/experiments/plume_data/right_plume_bounds.csv', names=col_names)
            df['y_position'] = self.experiment.windtunnel.heater_r.y_position
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        return df


class Timeavg_Plume(Plume):
    def __init__(self, experiment):
        super(Plume, self).__init__(experiment)

        self.data = self._load_plume_data()

    def in_plume(self, position):
        pass

    def show(self):
        pass

    def _load_plume_data(self):
        raise NotImplementedError


if __name__ == '__main__':
    windtunnel = Windtunnel('left')