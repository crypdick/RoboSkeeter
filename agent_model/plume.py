# -*- coding: utf-8 -*-
"""
Plume for dynamical agent model. Make a plume object, with methods such as temp_lookup

Created on Thu Apr 23 10:53:11 2015

@author: richard
"""

from scipy.spatial import cKDTree
import pandas as pd


#boundary = [2.0, 2.3, 0.15, -0.15, 1]  # these are real dims of our wind tunnel


def mkplume():
    plume = pd.read_csv('temperature_data/COMSOL2D_temp.csv')
    # grab just last timestep of the plume
    plume = plume.loc[:,['x', 'y', 't348']]
    plume.rename(columns={'t348': 'temp'}, inplace=True)
    # correct for model convention
    plume['y'] = plume.y.values - 0.127
    return plume


def mk_tree(plume):
    return cKDTree(plume[['x', 'y']])
#    return 0

#def templookup(coords):
#    return 0

class Plume():
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
        pass
        #TODO: plot plume
        
    

def main():
    plume = Plume()
    
    return plume
    
    

if __name__ == '__main__':
    plume = main()
    