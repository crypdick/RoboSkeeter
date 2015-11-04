__author__ = 'richard'

class Windtunnel():
    def __init__(self, test_condition):
        """
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)
        collision type:
            'elastic', 'crash'

        TODO: class  Wall. left wall right wall

        """
        self.walls = Walls()
        self.boundary = self.walls.boundary
        self.test_condition = test_condition

        self.left_heater = Heater('left')
        self.left_heater_loc = self.left_heater.xy_coords

        self.right_heater = Heater('right')
        self.right_heater_loc = self.right_heater.xy_coords

        self.on_heater = Heater(test_condition)
        if self.on_heater is not None:
            self.on_heater_loc = self.on_heater.xy_coords


class Walls():
    def __init__(self):
        # these are real dims of our wind tunnel
        self.left = 0.127
        self.right = -0.127
        self.upwind = 1.0
        self.downwind = 0.0
        self.ceiling = 0.254
        self.floor = 0.
        self.boundary = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]


class Heater():
    def __init__(self, side):
        ''' given {left, right, none, custom coords} place heater in the windtunnel

        Args:
        location

        returns [x,y, zmin, zmax, diam]
        '''
        self.zmin = 0.03800
        self.zmax = 0.11340
        self.diam = 0.01905
        self.side = side

        if side is None:
            self.xy_coords =  None
        elif side in "leftLeftLEFT":
            self.xy_coords =  [0.8651, -0.0507, self.zmin, self.zmax, self.diam]
        elif side in "rightRightRIGHT":
            self.xy_coords =   [0.8651, 0.0507, self.zmin, self.zmax, self.diam]
        else:
            raise Exception('invalid location type specified')


if __name__ == '__main__':
    windtunnel = Windtunnel('left')