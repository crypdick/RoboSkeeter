# -*- coding: utf-8 -*-
"""
Plume for dynamical agent model. Make a plume object, with methods such as temp_lookup

Created on Thu Apr 23 10:53:11 2015

@author: richard
"""

import numpy as np
import pandas as pd

from windtunnel import Walls


#boundary = [2.0, 2.3, 0.15, -0.15, 1]  # these are real dims of our wind tunnel


## depreciated in favor of simulated plume

#def mkplume():
#    ''' Genearate a plume based on a saved file.
#    
#    Input:
#    none
#    
#    Output:
#    Plume: (pandas dataframe, cols (x,y,temp))
#    '''
#    plume = pd.read_csv('temperature_data/LH_50_tempfield-10s.csv')
#    # grab just last timestep of the plume
#    plume = plume.loc[:,['x', 'y', 'T2 (K) @ t=30']]
#    # label pandas columns
#    plume.rename(columns={'T2 (K) @ t=30': 'temp'}, inplace=True)
#    # correction for model convention
#    plume['y'] = plume.y.values - 0.127
#    
#    return plume
#

# depreciated
#def mk_tree(plume):
#    ''' Creates a k-d tree map of the pandas Dataframe to look up temperatures
#    '''
#    return cKDTree(plume['x'])

class Plume():
    ''' Instantiates a plume object.
    
    Args:
    condition: {left|right|None}
    '''
    def __init__(self, condition):
        self.condition = condition
        self.data = self.load_plume_data()

        self.walls = Walls()

    def load_plume_data(self):
        col_names = ['fixme1', 'fixme2', 'fixme3']
        # col_names = ['x_position', 'y_position', 'radius']
        try:
            if self.condition in 'lLleftLeft':
                df = pd.read_csv('data/experiments/plume_data/left_plume_bounds.csv', names=col_names)
            elif self.condition in 'rightRight':
                df = pd.read_csv('data/experiments/plume_data/right_plume_bounds.csv', names=col_names)
        except TypeError:  # throwing None at in <str> freaks Python out
            df = pd.DataFrame.empty  # TODO: make plume that's 0 everywhere

        return df

    def in_plume(self, position):
        in_bounds, _ = self.walls.in_bounds(position)
        if in_bounds is False or self.condition is None:
            print("WARNING: sniffing outside of windtunnel bounds")
            inplume = 0
        else:
            x, y, z = position
            plume_plane = self._get_nearest_plume_plane(x)

            # divide by 3 to transform the ellipsoid space to a circle
            distance_from_center = ((y - plume_plane.y) ** 2 + (1 / 3 * (z - plume_plane.z)) ** 2) ** 0.5

            if distance_from_center <= plume_plane.small_radius:
                inplume = 1
            else:
                inplume = 0

        return inplume

    def _get_nearest_plume_plane(self, x_position):
        """given x position, find nearest plan"""
        closest_plume_index = (np.abs(self.data['position_x'] - x_position)).argmin()
        plume_plane = self.data.loc[closest_plume_index]

        return plume_plane

    def show(self):
        ''' show a plot of the plume object
        '''

        raise NotImplementedError
        
        



# depreciated      
#    def temp_lookup(self, position):
#        """Given position, find nearest temperature datum in our plume dataframe
#        using a k-d tree search.
#        
#        Input:
#        position: (list)
#            [xcoord, ycoord]
#        
#        Output:
#        tempearture: (float)
#            the nearest temperature
#        """
#        
#        distance, index = self.plume_kdTree.query(position) 
#        
#        return self.data.loc[index, 'temp']
#        
#
#    def show(self):
#        ''' show a plot of the plume object
#        '''
#        from matplotlib import pyplot as plt
#        
#        ax = self.data.plot(kind='hexbin', x='x', y='y', C='temp', reduce_C_function=np.max,
#                        gridsize=(20,60), vmax=297, title ="Temperature inside wind tunnel (K)", ylim=[-0.127, 0.127])
#        ax.set_aspect('equal')
#        ax.invert_yaxis()


if __name__ == '__main__':
    test_plume = Plume('l')
