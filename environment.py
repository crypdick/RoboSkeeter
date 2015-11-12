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

        self.heater = Heater(self.experimental_condition)
        self.heater.turn_on()

    def show(self):
        ax = pwt.plot_windtunnel()
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
    def __init__(self, experimental_condition):
        ''' given {left, right, none, custom coords} place heater in the windtunnel

        Args:
        location

        returns [x,y, zmin, zmax, diam]
        '''
        self.experimental_condition = experimental_condition

        if self.experimental_condition not in 'controlControlCONTROL':
            self.zmin = 0.03800
            self.zmax = 0.11340
            self.diam = 0.01905

            self.is_on = False

        self.x_position, self.y_position = self._set_xy_coords()

    def turn_on(self):
        self.is_on = True

    def _set_xy_coords(self):
        x_coord = -0.0507

        if self.experimental_condition in "leftLeftLEFT":
            y_coord = -0.0507
        elif self.experimental_condition in "rightRightRIGHT":
            y_coord = 0.0507
        elif self.experimental_condition in 'controlControlCONTROL':
            x_coord, y_coord = None, None
        else:
            raise Exception('invalid location type specified')

        return x_coord, y_coord


class Plume():
    ''' Instantiates a plume object.

    Args:
    condition: {left|right|None}
    '''

    def __init__(self, experiment):
        # useful aliases
        self.condition = experiment.condition
        self.heater = experiment.windtunnel.heater
        self.walls = experiment.windtunnel.walls

        self.data = self.load_plume_data()

    def load_plume_data(self):
        col_names = ['x_position', 'z_position', 'small_radius']

        if self.condition in 'controlControlCONTROL':
            return None  # TODO: make plume that's 0 everywhere
        elif self.condition in 'lLleftLeft':
            df = pd.read_csv('data/experiments/plume_data/left_plume_bounds.csv', names=col_names)
        elif self.condition in 'rightRight':
            df = pd.read_csv('data/experiments/plume_data/right_plume_bounds.csv', names=col_names)
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        df['y_position'] = self.heater.y_position

        return df

    def in_plume(self, position):
        in_bounds, _ = self.walls.in_bounds(position)
        if in_bounds is False or self.condition in 'controlControlCONTROL':
            # print("WARNING: sniffing outside of windtunnel bounds") #FIXME shouldn't be getting tripped
            inplume = 0
        else:
            x, y, z = position
            plume_plane = self._get_nearest_plume_plane(x)

            # divide by 3 to transform the ellipsoid space to a circle
            distance_from_center = ((y - plume_plane.y_position) ** 2 + (
            1 / 3 * (z - plume_plane.z_position)) ** 2) ** 0.5

            if distance_from_center <= plume_plane.small_radius:
                inplume = 1
            else:
                inplume = 0

        return inplume

    def _get_nearest_plume_plane(self, x_position):
        """given x position, find nearest plan"""
        closest_plume_index = (np.abs(self.data['x_position'] - x_position)).argmin()
        plume_plane = self.data.loc[closest_plume_index]

        return plume_plane

    def show(self):
        ax = scripts.plot_windtunnel.mk_3Dfig()
        scripts.plot_windtunnel.draw_windtunnel(ax)


if __name__ == '__main__':
    windtunnel = Windtunnel('left')