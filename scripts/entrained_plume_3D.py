# -*- coding: utf-8 -*-
"""
makes the plume from csv file
    
"""

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


def mkplume(condition='left'):
    ''' Genearate a plume based on a saved file.
    
    Input:
    none
    
    Output:
    Plume: (pandas dataframe, cols (x,y,temp))
    '''
    if condition in 'lLleftLeftLEFT':
        plume_data = pd.read_csv('plume_sim/left_plume_bounds.csv')
    elif condition in 'rRrightRightRIGHT':
        plume_data = pd.read_csv('plume_sim/right_plume_bounds.csv')
    elif condition in 'noneNonecontrolControl':
        plume_data = None
    # grab just last timestep of the plume
    print plume_data
#    plume_data = plume_data.loc[:,['x', 'y', 'small_radius']]
#    # label pandas columns
#    plume_data.rename(columns={'T2 (K) @ t=30': 'temp'}, inplace=True)
#    # correction for model convention
#    plume_data['y'] = plume.y.values - 0.127
    
    return plume_data


def mk_tree(plume):
    ''' Creates a k-d tree map of the pandas Dataframe to look up temperatures
    '''
    return cKDTree(plume[['x', 'y']])


class Plume():
    ''' Instantiates a plume object.
    '''
    def __init__(self):
        self.plume = mkplume()
        self.plume_kdTree = mk_tree(self.plume)
        
    def temp_lookup(self, position):
        """Given position, find nearest temperature datum in our plume dataframe
        using a k-d tree search.
        
        Input:
        position: (list)
            [xcoord, ycoord]
        
        Output:
        tempearture: (float)
            the nearest temperature
        """
        
        distance, index = self.plume_kdTree.query(position)
        
        return self.plume.loc[index, 'temp']
        
    def show(self):
        ''' show a plot of the plume object
        '''
        from matplotlib import pyplot as plt
        
        ax = self.plume.plot(kind='hexbin', x='x', y='y', C='temp', reduce_C_function=np.max,
                        gridsize=(20,60), vmax=297, title ="Temperature inside wind tunnel (K)", ylim=[-0.127, 0.127])
        ax.set_aspect('equal')
        ax.invert_yaxis()


def main():
    return Plume()
      

if __name__ == '__main__':
    plume = main()
    