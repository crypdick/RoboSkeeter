__author__ = 'richard'

class Windtunnel():
    def __init__(self, test_condition, collision_type='elastic'):
        """
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)

        """
        self.boundary = [0.0, 1.0, 0.127, -0.127, 0., 0.254]  # these are real dims of our wind tunnel
        self.test_condition = test_condition
        self.collision_type = collision_type

        self.left_heater = Heater('left')
        self.left_heater_loc = self.left_heater.xy_coords

        self.right_heater = Heater('right')
        self.right_heater_loc = self.right_heater.xy_coords

        self.on_heater = Heater(test_condition)
        if self.on_heater is not None:
            self.on_heater_loc = self.on_heater.xy_coords


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