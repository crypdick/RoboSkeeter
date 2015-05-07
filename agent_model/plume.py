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


def mkplume():
    ''' Genearate a plume based on a saved file.
    
    Input:
    none
    
    Output:
    Plume: (pandas dataframe, cols (x,y,temp))
    '''
    plume = pd.read_csv('temperature_data/LH_50_tempfield-10s.csv')
    # grab just last timestep of the plume
    plume = plume.loc[:,['x', 'y', 'T2 (K) @ t=30']]
    plume.rename(columns={'T2 (K) @ t=30': 'temp'}, inplace=True)
    # correct for model convention
    plume['y'] = plume.y.values - 0.127
    return plume


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
        ''' TODO plot the plume
        '''
        from matplotlib import pyplot as plt
        
        ax = self.plume.plot(kind='hexbin', x='x', y='y', C='temp', reduce_C_function=np.max,
                        gridsize=(20,60), vmax=297, title ="Temperature inside wind tunnel (K)", ylim=[-0.127, 0.127])
        ax.set_aspect('equal')
        ax.invert_yaxis()

        
    

def main():
    plume = Plume()
    
    return plume
    
    

if __name__ == '__main__':
    plume = main()
    