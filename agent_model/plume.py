# -*- coding: utf-8 -*-
"""
Plume for dynamical agent model

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
    

def main():
    plume = mkplume()
    plume_kdTree = mk_tree(plume)
    
    return plume, plume_kdTree
    
    

if __name__ == '__main__':
    plume, plume_kdTree = main()

#class Plume():
#    def __init__(self):
#        self.res = 0.08      # plumeresolution
#        x_ = np.arange(boundary[0]+self.res, boundary[1], self.res)
#        y_ = np.arange(boundary[2]-self.res, boundary[3]+self.res, -self.res)
#        z_ = [1.0]
#        
#        self.x, self.y, self.z = np.meshgrid(x_, y_, z_, indexing='ij')
##        # coords list for kdTree
##        self.coordinates_list = [(x,y) for x in self.X for y in self.Y]
##        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=False)
##        
##        self.ind = np.dstack((self.xx, self.yy))
##        self.Z = np.zeros_like(self.ind) + 1.0
##        
#        self.plume = np.dstack((self.x, self.y, self.z))
#        print self.plume
#        
#    def intensity(self, type="flat"):
##        self.Z = 0 * (self.xx + self.yy) + 0.1
#        if type == "flat":
#            self.Z = 0.0
#        
#    def show(self):
#        pass
#        fig, ax = plt.subplots(1)        
        
#        heatmap = ax.pcolormesh(self.x, self.y, self.z, cmap=cm.Oranges)
        
        
#        fig = plt.figure(0, figsize=(6,6))
#        ax = axes3d.Axes3D(fig)
#        x, y, z = plume.xx, plume.yy, plume.zz
#        ax.plot_surface(x , y , z)#, cmap=cm.hot)
#        
#        ax.set_xlim3d([boundary[0], boundary[1]])
#        ax.set_ylim3d([boundary[3], boundary[2]])
#        ax.set_zlim3d([0.0, boundary[4]])
#        ax.set_xlabel('X')
#        ax.set_ylabel('Y')
#        ax.set_zlabel('intensity')
#        ax.set_title("plume intensity in space")
        
#        plt.show()
    

#class Plume(object):
#    def __init__(self,):
#        """Our odor plume in x,y, z
#        Currenty 0 everywhere at all timesteps
#        TODO: feed plume a time parameter, which then calcs all the intensities 
#        """
#        self.res = 0.05      # plume is 1 cm resolution
#        self.X = np.arange(boundary[0], boundary[1], self.res)
#        self.Y = np.arange(boundary[2], boundary[3], -self.res)
#        # coords list for kdTree
#        self.coordinates_list = [(x,y) for x in self.X for y in self.Y]
#        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)
#        
#    def intensity(self, curr_time): #uses current time, not index
#        """given the timeindex, return plume intensity values
#        currently always returns 0
#        input curr_time
#        output plume at that frame
#        TODO: make vary over time"""
#        #odor intensity at x,y
#        self.zz = 0 * (self.xx + self.yy) + 0.1 # PLUME1 set to 1.1 everywhere
##==============================================================================
## # un-uniform plume      
##         foo = self.xx + self.yy
##         foo1 = (0 * foo[:len(foo)/2 -10])
##         foo2 = (0 * foo[len(foo)/2 -10:(len(foo)/2 + 20)]+ 3 )
##         foo3 = (0 * foo[len(foo)/2 + 20:])
##         self.zz = np.vstack((foo1,foo2,foo3)) # PLUME2 half 0 half 1.1
##         self.zz = self.zz.ravel()
#        #        self.plumexyz = [(x,y,z) for x in self.X for y in self.Y for z in self.zz.ravel()]
#        #TODO save 3d array to file so I don't need to recompute every time.
#
##==============================================================================
#        plume_curr = self.xx, self.yy, self.zz
#        
#        return plume_curr
#        
#    def find_nearest_intensity(self, loc):
#        """uses kd tree to find closest intensity coord to a given location
#        given a (x,y) location, return index of nearest intensity value
#        """
#        plumetree = spatial.cKDTree(self.coordinates_list)
#        distance, index = plumetree.query(loc)
#        
#        return self.coordinates_list[index]
#        
#    def intensity_val(self, plume_curr, location):
#        """
#        given a plume at a certain frame and x,y coords, give intensity at that coord
#        input plume, (x,y) coords
#        output: plume intensity at x,y
#        """
#        x, y = self.find_nearest_intensity(location)
#        intensitygrid = plume_curr[2] #THIS IS THE SOURCE OF THE ERROR!
#        try: 
#            return intensitygrid[x][y]
#        except IndexError:
#            print "mozzie sniffing outside the box"
#            return 0.0
#            
#    def show(self):
#        fig = plt.figure(0, figsize=(6,6))
#        ax = axes3d.Axes3D(fig)
#        x, y, z = plume.xx, plume.yy, plume.zz
#        ax.plot_surface(x , y , z)#, cmap=cm.hot)
#        
#        ax.set_xlim3d([boundary[0], boundary[1]])
#        ax.set_ylim3d([boundary[3], boundary[2]])
#        ax.set_zlim3d([0.0, boundary[4]])
#        ax.set_xlabel('X')
#        ax.set_ylabel('Y')
#        ax.set_zlabel('intensity')
#        ax.set_title("plume intensity in space")
#        
#        plt.show()
        
#            
#myplume = Plume()
#myplume.show()