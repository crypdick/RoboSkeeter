# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:14:50 2014

@author: richard

at each time index:
    generate intensity values x,y,z
    where is mozzie x y t
    intensity at x,y,t
    sensor neuron responds to stimulus.
        if it fires, saves the spike in [(time_index, 1)] and
        amplitude control neuron recompute y_targ
    mosquito moves.

"""
import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm


flight_dur = 100.0
timestep = 1

BOX_SIZE = (10.0,10.0,1.0) 

current_time_index = -1

class Plume(object):
    def __init__(self, time = 0):
        """Our odor plume in x,y, z
        Currenty 0 everywhere at all timesteps
        TODO: feed plume a time parameter, which then calcs all the intensities 
        """
        self.res = 100.0      #Split into 100 straight segments
        self.X = np.linspace(0, BOX_SIZE[0], self.res) #X locs of plume
        self.Y = np.linspace(0, BOX_SIZE[1],self.res) #Y of odor slices
        self.xx, self.yy = np.meshgrid(self.X, self.Y, sparse=True)
        self.zz = 0 * (self.xx + self.yy) # odor intensity at x,y, set to 0 everywhere
        #self.source = list(source) #Y/Z of the source
        #self.original = source #Initial source location
        #self.cross = np.zeros((2, len(self.X)))
    def intensity_val(self,location):
        #x, y = location
        #too
        return self.zz

      

class Mozzie(object):
    def __init__(self):
        self.pos =  0, 0
        print self.pos
    def where(self, time):
        """ TODO: make real"""
        return self.pos

class neuron(object):
    def __init__(self, tau_e = 1.2, spikethresh = 3.0):
        self.voltages = []
        self.spikes = []   
        self.spiketime_index = []
        pass

sensor_neuron = neuron()

def sensor_neuron(tau_e = 1.2, spikethresh = 3.0, plotting = 'off'): 
    '''evaluator leaky integrate and fire neuron. stimulus intensity -> cellular voltages -> spikes
    
    #TODO: feed in intensities
    '''
    voltages = []
    spikes = []
    time = 0
    for intensity in intensities: #TODO change to for time_curr in times:
        if len(voltages) == 0:
            voltages.append(0)
            spikes.append(0)
        else:
            if voltages[-1] > spikethresh: #crossing threshold discharges neuron
                voltages.append(0)
                spikes.append(0)
            else:
                voltage = (1/tau_e) * voltages[-1] + intensity #intensity at this timestep + weighted past intensity
                voltages.append(voltage)
                if voltage > spikethresh:
                    spikes.append(1)
                else:
                    spikes.append(0)
        time += 1
    if plotting == 'on':
        sensor_neuron_plotter(spikes, voltages)
    for timepoint, value in enumerate(spikes):
        if value == 1:
            spiketime_index.append(timepoint)
    return

#def agentflight(total_velo_max = 5, total_velo_min = 1, wind_velo = 0, y_offset_curr = 0, angular_velo = 0.1, tau_y = 1, plotting = 'off'):
#    """
#    plots the mosquito's path in space
#    input: total_velo_max, total_velo_min, wind_velo, y_offset_curr, angular_velo, flight_dur, timestep
#    output: times, agent_y_pos, x_velocities, y_velocities 
#    TODO: impliment agent_x_pos using velocity max and min
#    TODO: implement y offset, and wind velocity
#    TODO: make agent_pos list where each item is an (x,y) coord
#    TODO: make sensor_neuron control agent flight
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
    

def timestepper():
    times = np.arange(0, flight_dur, timestep)
    time_indexes = list(enumerate(times))
    for index in time_indexes:
        plume_now = plume.intensity_val(index[1]) #intensity value of plume at this time
        loc = mozzie.where(index[1])
        intensity_now = Plume.intensity_val(loc)
        print loc
        
        #return index[1] set global absolute time
#    while current_time_index < 
#    for time in times:
#        pass

if __name__ == "__main__":
    plume = Plume()
    mozzie = Mozzie()
    timestepper()
    plume_plotter(plume)
#    agentflightplotter() 