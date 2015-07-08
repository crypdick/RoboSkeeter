# -*- coding: utf-8 -*-
"""
Plume for dynamical agent model. Make a plume object, with methods such as temp_lookup

Created on Thu Apr 23 10:53:11 2015

@author: richard
"""

from scipy.spatial import cKDTree
import pandas as pd
import numpy as np


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
def load_simulated_plume(condition='left'):
    if condition in 'lLleftLeft':
        plume = pd.read_csv('plume_sim/left_plume_bounds.csv')
    if condition in 'rightRight':
        plume = pd.read_csv('plume_sim/right_plume_bounds.csv')
        
    return plume

# depreciated
#def mk_tree(plume):
#    ''' Creates a k-d tree map of the pandas Dataframe to look up temperatures
#    '''
#    return cKDTree(plume['x'])

class Plume():
    ''' Instantiates a plume object.
    '''
    def __init__(self):
        self.plume = load_simulated_plume(condition='left')
        self.x_vals = self.plume.x.values
#        self.plume_kdTree = mk_tree(self.plume) # depreciated      

    def find_nearest_plume_plane(self, x_val):
        closest_plume_index = (np.abs(self.x_vals - x_val)).argmin()
        plume_plane = self.plume.loc[closest_plume_index]
        
        return plume_plane


    def check_for_plume(self, position):
        x, y, z = position
        plume_plane = self.find_nearest_plume_plane(x)
        # divide by 3 to transform the ellipsoid space to a circle
        distance_from_center = ( (y - plume_plane.y)**2 + (1/3*(z - plume_plane.z)) ** 2) ** 0.5
        if distance_from_center < plume_plane.small_radius:
            return True
        else:
            return False
        
        



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
#        return self.plume.loc[index, 'temp']
#        
#
#    def show(self):
#        ''' show a plot of the plume object
#        '''
#        from matplotlib import pyplot as plt
#        
#        ax = self.plume.plot(kind='hexbin', x='x', y='y', C='temp', reduce_C_function=np.max,
#                        gridsize=(20,60), vmax=297, title ="Temperature inside wind tunnel (K)", ylim=[-0.127, 0.127])
#        ax.set_aspect('equal')
#        ax.invert_yaxis()


def main():
    return Plume()
      

if __name__ == '__main__':
    plume = main()
    