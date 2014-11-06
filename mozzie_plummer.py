# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:14:50 2014

@author: richard

at each time index:
    generate intensity values x,y,z
    mosquito moves to it's position in t
    where is mozzie x y t
    intensity at x,y,t
    sensor neuron responds to stimulus.
        if it fires, store the spike [(time_index, 1)] and
        amplitude control neuron recompute y_targ

"""
import numpy as np
from numpy import pi, sin, cos
from PIL import Image
import matplotlib.pyplot as plt

#==============================================================================
# TODAYS GOALS 
#
#  mess with parameteres
# 
#==============================================================================


class Plume(object):
    def __init__(self,):
        """Our odor plume in x,y, z
        plumefiles are all under vid1 folder
        """
        self.X = range(0, BOX_SIZE[0]) #X locs of plume
        self.Y = range(0, BOX_SIZE[1]) #Y of odor slices
        self.coordinates_list = [(x,y) for x in self.X for y in self.Y]
        self.plumefiles = [filename for filename in glob('./vid1/*')]
    def current_plume(self, curr_time_index):
        """given specific time, what is our plume?
        The first section are static plume images;
        the second section makes use of the time index to look up the corresponding
            video frame.
        """
#======= DEBUGGING/STATIC SAMPLE PLUMES ======================================
#        imgdir = "./example_vids/fullstim.png"
#        imgdir = "./example_vids/nostim.png"
        imgdir = "./example_vids/diagplume.png"
#        imgdir = "./example_vids/topplume.png" #boring
#        imgdir = "./example_vids/midplume.png" #boring
#        imgdir = "./example_vids/gaussian.png"
#        imgdir = "./example_vids/realplume.png"
        img =Image.open(imgdir).convert('L')
        return img
#========Moving plume===========================================  
#        try:
#            img = Image.open(self.plumefiles[curr_time_index]).convert('L')
#        except IndexError:
#            print """Flight duration set longer than available plume movie! Stopping
#            loop at timestep %s""" % curr_time_index
#            return None
#        else:
#            return img
        
#==============================================================================
#find nearest intensity. not required for now because getpixel(x,y) already rounds
#     def find_nearest_intensity(self,loc):
#         """uses kd tree to find closest intensity coord to a given location
#         given a (x,y) location, return index of nearest intensity value
#         """
#         plumetree = spatial.cKDTree(self.coordinates_list)
#         distance, index = plumetree.query(loc)
#         print "xy2PIL", loc, "nearest intensity", self.coordinates_list[index]
#         return self.coordinates_list[index]
#==============================================================================
    def coord2PILarray(self,xycoords):
        """takes coords in our standard x,y coord system and returns them
        in the coordinate system for the PIL package, which defines (0,0) as being
        the top left corner"""
        x, y = xycoords
        y = abs(y - BOX_SIZE[1]) #correcting y's for the coordinate axis of the PIL array
#==============================================================================
#        ##not needed unless we use kd tree function
#         nearest_x, nearest_y = self.find_nearest_intensity((x,y))
#         return nearest_x, nearest_y
#==============================================================================
        return x,y
    def intensity_val(self,plume_curr,coord):
        """given a plume img and a coord, returns the intensity value at that pixel.
        converts coordinate axis to the corresponding one on the PIL array first
        """
        x,y = self.coord2PILarray(coord) #convert first
        try: 
             return plume_curr.getpixel((x,y))  
        except IndexError:
             print "mozzie sniffing outside the box"
             return 0.0
        

class Mozzie(object):
    """our brave mosquito, moving in 2D
    input: total_velo_max, total_velo_min, wind_velo, y_offset_curr, angular_velo, flight_dur, timestep_size
    output: times, agent_y_pos, x_velocities, y_velocities 
    TODO: make sensor_neuron control agent flight
    """
    def __init__(self,total_velo_max = 20, wind_velocity = 0, y_offset_curr = 0, angular_velo = pi / 15): #total velo max originally set to 6.6
        self.total_velo_max = total_velo_max
        self.total_velo_min = total_velo_max / 5
        self.angular_velo = angular_velo
        self.wind_velo = wind_velocity
        self.loc_curr =  0, BOX_SIZE[1]/2
        self.loc_history = {0 : self.loc_curr} #dict of locations
        self.loc_list = []
        self.y_velocities = {}
        self.x_velocities = {}
        self.time_prev = 0
    def move(self,time_curr):
        """move the mosquito in a cast, add to location history
        input: time in secs
        output: none (location history updates)
        TODO: impliment agent_x_pos using velocity max and min
        TODO: wind velocity
        """
        y_pos_curr = amplitude_neuron.amplitude_curr * sin (self.angular_velo * time_curr) + amplitude_neuron.y_offset_curr + BOX_SIZE[1]/2 #BOXSIZE to convert to the standard coordinate axis
        garbage, y_pos_prev = self.loc_history[self.time_prev] #we only want the second value
        y_velocity_curr = (y_pos_curr - y_pos_prev) / timestep_size
        self.y_velocities[time_curr] = y_velocity_curr
#        ## TODO: make x_velocities as in Sharri's
        self.x_velocities[time_curr] = (time_curr - self.time_prev)/ timestep_size
        self.loc_curr = time_curr, y_pos_curr #TODO: make x coord not the time!
        self.time_prev = time_curr
        self.loc_history[time_curr] = self.loc_curr
    def where(self, time):
        """ given time, where was the mozzie?
        input: time in seconds
        output: x,y coords
        TODO: make real"""
        where = self.loc_history[time]
        return where


def plume_plotter(plume, plotting = False):
     if plotting == False:
         pass
     else:
         debugplume.show()

def mozzie_plotter(plotting = False):
    if plotting == False:
        pass
    else:
        fig = plt.figure(1, figsize=(6,6))
        plt.xlabel('time')
        plt.ylabel('Y position')
        plt.title("Flying mosquito")  
        
        #plot the mozzie
        location_times, locations = mozzie.loc_history.keys(), mozzie.loc_history.values()
        x = [float(xycoord[0]) for xycoord in mozzie.loc_history.values() ] #THIS IS CURRENTLY TIME!!
        y = [float(xycoord[1]) for xycoord in mozzie.loc_history.values() ]
        plt.plot(x, y, 'k',label= 'agent y_pos over time' )
        
        #plot box boundaries
        plt.plot(location_times,len(location_times)*[BOX_SIZE[1]])
        plt.plot(location_times,len(location_times)*[0])
        
        #Plot sniff spots
        try: 
             sniffx, sniffy = zip(*sniffspots)
             plt.scatter(sniffx, sniffy,color='orange', marker="^", label='spike locations')  
        except ValueError: #if no sniffs, no sniff scatterplot
             pass  
        
        #plot velocity calculations
#        y_velocity_times, y_velocities = mozzie.y_velocities.keys(), mozzie.y_velocities.values()
#        x_velocities_times, x_velocities = mozzie.x_velocities.keys(), mozzie.x_velocities.values()
#        plt.plot(y_velocity_times, y_velocities,'b',label= 'y velocity over t')
#        plt.plot(x_velocities_times,x_velocities,'r',label= 'x velocity over t')
        
        plt.legend()
        plt.show()


class Neuron(object):
    def __init__(self, tau_e = 1.2, spikethresh = 100.0):
        self.spikethresh = spikethresh
        self.tau_e = tau_e
        self.voltage_history = {}
        self.spike_history = {} 
        self.spiketime_index = {}
        self.time_prev = 0
        #if spiking, return true

class Sensor_neuron(Neuron):
    def spiker(self,time, intensity_now):
        '''evaluator leaky integrate and fire neuron. stimulus intensity -> cellular voltages -> spikes
        input time (sec) and intensity
        updates spiking history and voltage history
        output spiking True and timestep
        '''
        if len(self.voltage_history) == 0: 
            """at very first timestep, start neuron fresh
            TODO: could this be avoided? remove self.tau weighted computation"""
            self.voltage_history[time] = 0.0
            self.spike_history[time] = 0
            self.time_prev = time
        else:
            voltage_prev = self.voltage_history[self.time_prev]
            if voltage_prev > self.spikethresh: #crossing threshold at prev timestep discharges neuron voltage
                self.voltage_history[time] = 0
                self.spike_history[time] = 1
                self.spiketime_index[time] = 1
                sniffspots.append(mozzie.where(time))
                self.time_prev = time
                return "spike!"
            else:
                #updating voltage
                self.voltage = (1/self.tau_e) * voltage_prev + intensity_now #intensity at this timestep + weighted past intensity
                self.voltage_history[time] = self.voltage
                self.spike_history[time] = 0
                self.time_prev = time        

def eval_neuron_plotter(plotting = False):
    """plots soma voltage and spike events """
    if plotting == False:
        pass
    else:
        fig = plt.figure(2)
        plt.xlabel('time')
        plt.ylabel('voltage')
        plt.title("Neuron voltage and spikes")   
        
        voltagetimes, voltages = sensor_neuron.voltage_history.keys(), sensor_neuron.voltage_history.values()
        spiketimes, spikes = sensor_neuron.spike_history.keys(), sensor_neuron.spike_history.values()
        spikes = [spike * 50 for spike in spikes] #make spikes visible
        plt.plot(voltagetimes, voltages, 'r',label='soma voltage')
        plt.plot(spiketimes, spikes, 'b', label= 'spikes', marker=".")
        
        plt.legend()
        plt.show()

class Amplitude_neuron(Neuron):
    """Our amplitude control neuron
    if the upstream sensor neuron fires in this timestep, it recomputes y_aim (=y_targ), 
    and amplitude val.
    else, decay teleporter
    """
    def __init__(self):
        self.time_prev = 0.0
        self.tau_y = 10
        self.y_aim = BOX_SIZE[1]/2 #aim for centerline to start
        self.y_aim_history = {0.0:self.y_aim} #unncessary?
        self.y_offset_curr = 0.0
        self.y_offset_history = {0.0:self.y_offset_curr}
        self.angular_velo = mozzie.angular_velo
        #TODO: understand amp max func
        self.amplitude_max = (2**0.5 / self.angular_velo) * (((mozzie.total_velo_max + mozzie.wind_velo) **2) - ((mozzie.total_velo_min + mozzie.wind_velo)**2))**0.5
        self.amplitude_curr = self.amplitude_max
    def y_aimer(self,time):
        """recompute y_aim to whichever y the mozzie is currently at.
        y_aim is where center axis that the mosquito gravitates towards using
        the y_offsetter fxn.
        only invoked by timestepper when the sensor neuron is spiking
        
        the idea is that the mosquito will stay inside the plume"""
        x_curr, y_curr = mozzie.where(time)
        self.y_aim = y_curr
        self.y_aim_history[time] = self.y_aim #unncessary?
    def y_offsetter(self,time):
        """teleports the mosquito towards y_aim
        """
        self.y_offset_curr = (self.y_aim - self.y_offset_history[self.time_prev]) / self.tau_y
        self.y_offset_history[time] = self.y_offset_curr
        self.time_prev = time
    def amplitude_controller(self):
        """shrinks amplitude when the mosquito is in the plume, and if there's no
        spiking in the sensor neuron, enlarges the amp until it's at amplitude_curr = amplitude_max
        TODO implement!
        TODO add vwind
        TODO constrain mosquito to the box no matter what"""
        pass
        

def timestepper():
    """the main function driving all the code. divides the flight_dur into timesteps,
    then for each timestep goes through the entire sequence of events.
    what is the plume at this time?
    where is the mozzie atm? what's the intensity there?
    update the sensor neuron based on that intensity.
    did it fire? if so, recompute y_aim.
    compute for next timestep: how much we will teleport the mozzie and the new amplitude
    """
    times = np.arange(0, flight_dur, timestep_size)
    time_indexes = list(enumerate(times))
    for time in time_indexes:
        plume_curr = plume.current_plume(time[0]) #get current plume. feeding [0] to get time index.
        if plume_curr is None: 
            #in case our flighttime > plume animation time
            break
        else:
            mozzie.move(time[1])
            loc = mozzie.where(time[1])
            intensity_now = plume.intensity_val(plume_curr,loc)
            if sensor_neuron.spiker(time[1], intensity_now) == "spike!": #if sensor neuron spikes
                amplitude_neuron.y_aimer(time[1])
            amplitude_neuron.y_offsetter(time[1])
            amplitude_neuron.amplitude_controller()
        
            

if __name__ == "__main__":
    flight_dur = 550.0 #make 720
    timestep_size = 1.0
    sniffspots = [] #this will be used later to scatterplot the locations where the eval neuron spiked

    BOX_SIZE = (720,420,255)
    sensor_neuron = Sensor_neuron()
    mozzie = Mozzie()
    amplitude_neuron = Amplitude_neuron()
    plume = Plume()
    timestepper()
#    plume_plotter(plume, plotting = True)
    eval_neuron_plotter(plotting = True)
    mozzie_plotter(plotting = True)
    
#==============================================================================
#     FOR DEBUGGING
#    debugplume = plume.current_plume(0) 
#    plume_plotter(debugplume, plotting = True)
#     
#     
# #==============================================================================
# #     example debug commands
# #     print plume.intensity_val(debugplume,(0,0))
# #     
# #==============================================================================
#==============================================================================