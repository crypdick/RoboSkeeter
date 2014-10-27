# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:14:50 2014

@author: richard
"""
import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm


flight_dur = 100.0
timestep = 0.5
times = np.arange(0, flight_dur, timestep)
time_indexes = list(enumerate(times))

BOX_SIZE = (10.0,10.0,1.0) 


class plume(object):
    def __init__(self):
        """A time-evolving odorant for our little guy to flap around in"""
        self.res = 100.0      #Split into 100 straight segments
        self.X = np.linspace(0, BOX_SIZE[0], self.res) #X locs of plume
        self.Y = np.linspace(0, BOX_SIZE[1],self.res) #Y of odor slices
        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)
        self.zz = 0 * (self.xx + self.yy) # odor intensity at x,y, set to 0 everywhere
        #self.source = list(source) #Y/Z of the source
        #self.original = source #Initial source location
        #self.cross = np.zeros((2, len(self.X)))

# Set up a plume
plume = plume()
      
def plume_plotter(plume):
    # Set up a figure
    fig = plt.figure(0, figsize=(6,6))
    ax = axes3d.Axes3D(fig)
    x, y, z = plume.xx, plume.yy, plume.zz
    ax.plot_wireframe(x , y , z, rstride=10, cstride=10)
    ax.set_xlim3d([0.0, BOX_SIZE[0]])
    ax.set_ylim3d([0.0, BOX_SIZE[1]])
    ax.set_zlim3d([0.0, BOX_SIZE[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('intensity')
    ax.set_title("robomozzie")
    plt.show()
    
plume_plotter(plume)

#make this fluctuating
#intensities = ([0] * 5 ) + ([1] * 10 ) + ([0] * 20 ) + ([5] * 15) + ([1] * 150 )
#intensities_indexed = list(enumerate(intensities))
#print intensities_indexed
#def intensity(time_index):
#    pass 

##intensities = ([0] * 5 ) + ([1] * 10 ) + ([0] * 20 ) + ([5] * 15) + ([1] * 150 )
#spiketime_index = []
#
#class mozzie(object):
#    pass
#
#class neuron(object):
#    pass
#
#sensor_neuron = neuron()



#def agentflight(total_velo_max = 5, total_velo_min = 1, wind_velo = 0, y_offset_curr = 0, angular_velo = 0.1, tau_y = 1, plotting = 'off'):
#    """
#    plots the mosquito's path in space
#    input: total_velo_max, total_velo_min, wind_velo, y_offset_curr, angular_velo, flight_dur, timestep
#    output: times, agent_y_pos, x_velocities, y_velocities 
#    TODO: impliment agent_x_pos using velocity max and min
#    TODO: implement y offset, and wind velocity
#    TODO: make agent_pos list where each item is an (x,y) coord
#    TODO: make eval_neuron control agent flight
#    """
#    amplitude_max = 10 # TODO: impliment the amplitude fxn
#    agent_y_pos = []
#    y_velocities = []
#    x_velocities = []
#    for time_curr in times:
#        y_pos_curr = amplitude_max * sin (angular_velo * time_curr) #+ y_offset_curr #disabled
#        agent_y_pos.append(y_pos_curr)
#        y_offset_prev = y_offset_curr #test this with print statements
#        if time_curr == 0:
#            time_prev = 0
#            y_pos_prev = 0
#            time_index = 0
#        y_velocity = (y_pos_curr - y_pos_prev) / timestep
#        y_velocities.append(y_velocity)
#        ## TODO: make x_velocities as in Sharri's
#        x_velocities.append((time_curr - time_prev)/ timestep)
#        if time_index in spiketime_index:         # if we spiked at this time_curr, set y_aim = y_post_cur
#            y_aim = y_pos_curr
#            y_offset_curr = (y_aim - y_offset_prev) / tau_y
#        time_prev = time_curr
#        y_pos_prev = y_pos_curr
#        time_index += 1
#    if plotting == "on":
#        agentflightplotter(agent_y_pos, x_velocities, y_velocities)
#    return times, agent_y_pos, x_velocities, y_velocities    

    
#def agentflightplotter(agent_y_pos, x_velocities, y_velocities):
#    """
#    TODO: animate this, using def update_mozzie and FuncAnimation
#    """
#    fig = plt.figure(2)
#    plot(times,agent_y_pos,'k',label= 'agent y_pos over time' )
#    plot(times,y_velocities, 'b',label= 'y velocity')
#    plot(times,x_velocities, 'r',label= 'x velocity')    
#    xlabel('time')
#    ylabel('Y position')
#    legend()
#    title("Flying mosquito")
#    plt.show()

if __name__ == "__main__":
    #timestepper()
#    agentflightplotter() 
    pass