__author__ = 'richard'

class Windtunnel():
    def __init__(self, test_condition, collision='elastic'):
        """
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)

        """
        self.boundary = [0.0, 1.0, 0.127, -0.127, 0., 0.254]  # these are real dims of our wind tunnel
        self.test_condition = test_condition
        # place heater
        self.place_heater(test_condition)
        self.collision = collision

    def place_heater(self, test_condition):
        ''' given {left, right, none, custom coords} place heater in the windtunnel

        Args:
        location

        returns [x,y, zmin, zmax, diam]
        '''
        zmin = 0.03800
        zmax = 0.11340
        diam = 0.01905
        if test_condition is None:
            self.heater_location = None
        elif test_condition in "leftLeftLEFT":
            self.heater_location = [0.8651, -0.0507, zmin, zmax, diam, 'left']
        elif test_condition in "rightRightRIGHT":
            self.heater_location =  [0.8651, 0.0507, zmin, zmax, diam, 'right']
        elif type(test_condition) is list:
            self.heater_location = test_condition
        else:
            raise Exception('invalid location type specified')


if __name__ == '__main__':
    windtunnel = Windtunnel('left')