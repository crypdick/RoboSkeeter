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
        #self.source = list(source) #Y/Z of the source
        #self.original = source #Initial source location
        #self.cross = np.zeros((2, len(self.X)))
    def current_plume(self,curr_time): #uses current time, not index
        """given the timeindex, return plume intensity values
        currently always returns 0
        input curr_time
        output plume at that frame
        TODO: make vary over time"""
        self.zz = 0 * (self.xx + self.yy) # odor intensity at x,y, set to 0 everywhere
        plume_curr = self.xx, self.yy, self.zz
        return plume_curr
    def intensity_val(self, plume_curr, location):
        """
        given a plume at a certain frame and x,y coords, give intensity at that coord
        input plume, (x,y) coords
        output: plume intensity at x,y
        """
        x, y = location
        intensitygrid = plume_curr[2]
        return intensitygrid[x][y]

class Mozzie(object):
    """our brave mosquito
    """
    def __init__(self):
        self.pos =  0, 0
        self.loc_history = []
    def where(self, time):
        """ where the mozzie is right now
        input: time in seconds
        output: x,y coords
        TODO: make real"""
        return self.pos
    def move(self,time):
        """move the mosquito in a cast, add to location history
        input: time in secs
        output: none (location history updates)
        TODO: put in sin equations
        """
        self.loc_history.append((time,self.pos))
    # def _xspeedcalc
        

class Neuron(object):
    def __init__(self, tau_e = 1.2, spikethresh = 3.0):
        self.spikethresh = spikethresh
        self.tau_e = tau_e
        self.voltage_history = []
        self.spike_history = []   
        self.spiketime_index = []
        #if spiking, return true

class Sensor_neuron(Neuron):     
    def spiker(self,time, intensity_now):
        '''evaluator leaky integrate and fire neuron. stimulus intensity -> cellular voltages -> spikes
        
        TODO: feed in intensities
        '''
        if len(self.voltage_history) == 0:
            self.voltage_history.append(0)
           # spike_history.append(0)
        else:
            if self.voltage_history[-1] > self.spikethresh: #crossing threshold discharges neuron
                self.voltage_history.append(0)
                self.spike_history.append(0)
            else:
                self.voltage = (1/self.tau_e) * self.voltage_history[-1] + intensity_now #intensity at this timestep + weighted past intensity
                self.voltage_history.append(self.voltage)
                if self.voltage > self.spikethresh:
                    self.spike_history.append(1)
                else:
                    self.spike_history.append(0)
        for timepoint, value in enumerate(self.spike_history):
            if value == 1:
                spiketime_index.append(timepoint)

def sensor_neuron_old(tau_e = 1.2, spikethresh = 3.0, plotting = 'off'): 
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

def plume_plotter(plume, plotting = False):
    if plotting == False:
        pass
    else:
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
    for time in time_indexes:
        plume_curr = plume.current_plume(time[1]) #intensity value of plume at this time
        loc = mozzie.where(time[1])
        intensity_now = plume.intensity_val(plume_curr,loc)
        if sensor_neuron.spiker(time[1], intensity_now) == True: #if sensor neuron spikes
            amplitude_neuron(time[1])
        mozzie.move(time[1])
            

if __name__ == "__main__":
    plume = Plume()
    sensor_neuron = Sensor_neuron()
    amplitude_neuron = Neuron()
    mozzie = Mozzie()
    timestepper()
    plume_plotter(plume, plotting = False)
#    agentflightplotter() 