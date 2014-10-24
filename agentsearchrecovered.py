# -*- coding: utf-8 -*-

#def agentsearchrecovered(thresh=10, tau_e=24.65,tau_a=92.21,s=13.25,y_0,drw='off'):
#    """Agent Search Recovered
#    Runs agent model search simulation using brightness levels of pixels of a
#    movie as stimulus input. Agent navigates using response of to LIF neurons
#    an evaluator (E), and an actuator (A).
#    INPUT:
#    thresh : detection threshold of the evaluator LIF neuron
#    tau_e : decay constant (history) of the evaluator LIF neuron
#    tau_a : decay constnat (history) of the actuator LIF neuron
#    s : scale of amplitude decrease, controlled by actuator neuron
#    y_0 : initial y position of agent
#    drw : (default 'off') string input to draw all process inputs. Enter 'on' to draw.
#    
#    OUTPUT:
#    psuccess - percent successful trials
#    st - average search time (length of search)
#    sst - average search time for successful trials
#    xvec - agent path, x dimension
#    yvec - agent path, y dimension
#    da - actuator input (evaluator spike train convolved with Gaussian)
#    a - actuator output (to check for egregious error)
#    
#    Suggested input values are: 
#    thresh = 10
#    tau_e = 24.65
#    tau_a = 92.21
#    s = 13.25
#    """
#    return none
from pylab import *

flight_dur = 100.0
timestep = .5
times = numpy.arange(0, flight_dur, timestep)
intensities = ([0] * 5 ) + ([1] * 10 ) + ([0] * 20 ) + ([5] * 15) + ([1] * 150 )
spiketime_index = []

def eval_neuron(tau_e = 1.2, spikethresh = 3.0, plotting = 'off'): 
    '''evaluator leaky integrate and fire neuron. stimulus intensity -> cellular voltages -> spikes
    
    #TODO: feed in intensities
    '''
    voltages = []
    spikes = []
    time = 0
    for intensity in intensities: #TODO change to for time_curr in times:
        time += 1
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
    if plotting == 'on':
        eval_neuron_plotter(spikes, voltages)
    for timepoint, value in enumerate(spikes):
        if value == 1:
            spiketime_index.append(timepoint)
    return


        
def agentflight(total_velo_max = 5, total_velo_min = 1, wind_velo = 0, y_offset_curr = 0, angular_velo = 0.1, tau_y = 1, plotting = 'off'):
    """
    plots the mosquito's path in space
    input: total_velo_max, total_velo_min, wind_velo, y_offset_curr, angular_velo, flight_dur, timestep
    output: times, agent_y_pos, x_velocities, y_velocities 
    TODO: impliment agent_x_pos using velocity max and min
    TODO: implement y offset, and wind velocity
    TODO: make agent_pos list where each item is an (x,y) coord
    TODO: make eval_neuron control agent flight
    """
    amplitude_max = 10 # TODO: impliment the amplitude fxn
    agent_y_pos = []
    y_velocities = []
    x_velocities = []
    for time_curr in times:
        y_pos_curr = amplitude_max * sin (angular_velo * time_curr) + y_offset_curr
        agent_y_pos.append(y_pos_curr)
        y_offset_prev = y_offset_curr #test this with print statements
        if time_curr == 0:
            time_prev = 0
            y_pos_prev = 0
            time_index = 0
        y_velocity = (y_pos_curr - y_pos_prev) / timestep
        y_velocities.append(y_velocity)
        ## TODO: make x_velocities as in Sharri's
        x_velocities.append((time_curr - time_prev)/ timestep)
        if time_index in spiketime_index:         # if we spiked at this time_curr, set y_aim = y_post_cur
            y_aim = y_pos_curr
            y_offset_curr = (y_aim - y_offset_prev) / tau_y
        time_prev = time_curr
        y_pos_prev = y_pos_curr
        time_index += 1
    if plotting == "on":
        agentflightplotter(x_velocities, y_velocities)
    return times, agent_y_pos, x_velocities, y_velocities    
    
def agentflightplotter(x_velocities, y_velocities):
        figure(2)
        plot(times,agent_y_pos,'k',label= 'agent y_pos over time' )
        plot(times,y_velocities, 'b',label= 'y velocity')
        plot(times,x_velocities, 'r',label= 'x velocity')    
        xlabel('time')
        ylabel('Y position')
        legend()
        title("Flying mosquito")

def eval_neuron_plotter(spikes, voltages):
        y = max(voltages)    
        figure(1)
        plot(times, voltages, 'r',label='soma voltage')
        plot(times, spikes, 'b', label= 'spikes')
        xlabel('time')
        ylabel('voltage')
        legend()
        title("Neuron voltage and spikes")


if __name__ == "__main__":
    eval_neuron(plotting = 'on')
    times,agent_y_pos, x_velocities, y_velocities = agentflight(plotting = 'on')