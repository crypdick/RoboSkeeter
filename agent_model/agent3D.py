# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard
"""

import numpy as np
from math import atan2
import baseline_driving_forces3D
import stim_biasF3D
import plume3D
import trajectory3D
import windtunnel
import sys
import pandas as pd


def place_agent(agent_pos='downwind_plane'):
    ''' puts the agent in an initial position, usually within the bounds of the
    cage

    Options: center [of the cage], [anywhere in the] cage, [the cage] door, or anywhere in the plane at x=.1 meters
    '''
    if type(agent_pos) is list:
        return agent_pos
    if agent_pos == "center":
        return [0.1524, 0., 0.]  # center of box
        # FIXME cage height
    if agent_pos == "cage":  # bounds of cage
        return [np.random.uniform(0.1143, 0.1909), np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)]
        # FIXME cage height
    if agent_pos == "door":  # start trajectories when they exit the front door
        return [0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)]
        # FIXME cage height
    if agent_pos == 'downwind_plane':
        return [0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)]
    else:
        raise Exception('invalid agent position specified')




def solve_heading(velo_x, velo_y):
    theta = atan2(velo_y, velo_x)
    return theta*180/np.pi


# def fly_wrapper(agent_obj, args, traj_count):
#     """wrapper fxn for fly_single to use for multithreading
#     """
#     vectors_object = agent_obj._fly_single(*args)
#     setattr(vectors_object, 'trajectory_num', traj_count)
#     df = pd.DataFrame(vars(vectors_object))
#
#
#     return df


def attraction_basin(k, pos, y_spring_center=0.12, z_spring_center=0.2):
    """given y position, determine if the agent is currently flighing in the left or right half of the windtunnel.
    then, figures out which spring center to use,
     and returns the spring force.

    TODO: solve for correct spring centers
    TODO: make separate k for y, z?
    """
    x_pos, y_pos, z_pos = pos
    if y_pos >= 0:
        y_sign = 1
    else:
        y_sign = -1
    return np.array([
        0.,
        k * ((y_sign * y_spring_center) -  y_pos),
        k * (z_spring_center -  z_pos)
        ])



class Agent():
    """Generate agent (our simulated mosquito) which can fly.

    TODO: separate windtunnel environment into its own class

    Args:
        Trajectory_object: (trajectory object)
            pandas dataframe which we store all our simulated trajectories and their kinematics
        Plume_object: (plume object)
            the temperature data for our thermal, convective plume
        windtunnel_object: (windtunnel object)
            our virtual wind tunnel
        agent_position: (list/array, "cage", "center")
            how to pick the initial position coordinates of a flight
        v0_stdev: (float)
            stdev of initial velocity distribution (default: experimentally observed v0, fit to a normal distribution)
        heater: (list/array, "left", "right", or "None")
            the heater condition: left/right heater turned on (set to None if no heater is on)
        Tmax: (float)
            max time an agent is allowed to fly
            (data: <control flight duration> = 4.4131 +- 4.4096)  # TODO: recheck when Sharri's new data comes in
        dt: (float)
            width of our time steps
        k: (float)
            spring constant for our wall attraction flow
        beta: (float)
            damping force (kg/s). Undamped = 0, Critical damping = 1,t
            NOTE: if beta is too big, the agent accelerates to infinity!
        biasF_scale: (float)
            parameter to scale the main driving force
        wtf: (float)
            magnitude of upwind bias force   # TODO units
        

    All args are in SI units and based on behavioral data:

    

    Returns:
        Agent object

    """
    def __init__(
        self,
        Trajectory_object,
        Plume_object,
        Windtunnel_object,
        agent_pos='door',
        v0_stdev=0.01,
        Tmax=15.,
        dt=0.01,
        heater='left',
        beta=1e-5,
        randomF_strength=4e-06,
        windF_strength=5e-06,
        stimF_str=1e-4,
        bounded=True,
        mass = 3e-6, #3.0e-6 # 2.88e-6  # mass (kg) =2.88 mg,
        k=0.):
        
        self.metadata = {}

        # population weight data: 2.88 +- 0.35mg
        self.metadata['mass'] = mass
        self.metadata['time_max'] = Tmax
        self.metadata['time bindwidth'] = dt
        self.metadata['stimF_strength'] = stimF_str

        self.metadata['boundary'] = Windtunnel_object.boundary
        self.metadata['heater_position'] = Windtunnel_object.test_condition
        self.metadata['initial_position'] = agent_pos
        self.metadata['initial_velo_stdev'] = v0_stdev
        self.metadata['k'] = k
        self.metadata['beta'] = beta
        self.metadata['randomF_strength'] = randomF_strength
        self.metadata['wtF'] = windF_strength
        # for stats, later
        self.metadata['time_target_find_avg'] = []
        self.metadata['total_finds'] = 0
        self.metadata['target_found'] = [False]  # TODO: shouldn't this be attached to Trajectories() instead?
        self.metadata['time_to_target_find'] = [np.nan] # TODO: make sure these lists are concating correctly
        
        self.metadata['kinematic_vals'] = ['velocity', 'acceleration']
        self.metadata['forces'] = ['totalF', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
        
        
        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.metadata['turn_threshold'] = 433.5         
        
        self.trajectory_obj = Trajectory_object
        self.plume_obj = Plume_object
        
        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.metadata['boundary'])


    def fly(self, total_trajectories=1, verbose=True):
        ''' iterates self._fly_single() total_trajectories times
        '''
        traj_count = 0
        args = (self.metadata['time bindwidth'], self.metadata['mass'], self.metadata['boundary'])
        df_list = []
        while traj_count < total_trajectories:
            if verbose is True:
                sys.stdout.write("\rTrajectory {}/{}".format(traj_count+1, total_trajectories))
                sys.stdout.flush()


            kinematics_dict = self._fly_single(*args)
            kinematics_dict['trajectory_num'] = traj_count  # enumerate the trajectories
            df = pd.DataFrame(kinematics_dict)

            df_list.append(df)
            
            traj_count += 1
            if traj_count == total_trajectories:
                if verbose is True:
                    sys.stdout.write("\rSimulations finished. Performing deep magic.")
                    sys.stdout.flush()
        

        
        self.metadata['total_trajectories'] = total_trajectories
        self.trajectory_obj.add_agent_info(self.metadata)
        self.trajectory_obj.load_ensemble(df_list)  # concatinate all the dataframes at once instead of one at a
                                                          # time for performance boost.
        # concluding stats
#        self.trajectory_obj.add_agent_info({'time_target_find_avg': trajectory3D.T_find_stats(self.trajectory_obj.agent_info['time_to_target_find'])})
    
    def _fly_single(self, dt, m, boundary, bounded=True):
        """Generate a single trajectory using our model.
    
        First put everything into np arrays stored inside of a dictionary
        """
        BOUNCE = "elastic"
        PLUME_TRIGGER_TIME = int(1.0/dt)
        tsi_max = int(np.ceil(self.metadata['time_max'] / dt))  # N bins/maximum time step
        
        # V for vector
        # retrieve dict w/ vars(V)
        # V = Object()
        V = {}
        
        for name in self.metadata['kinematic_vals']+self.metadata['forces']:
            # adds attribute to V
            for ext in ['_x', '_y', '_z', '_xy_theta', '_xy_mag']:
                # setattr(V, name+ext, np.full(tsi_max, np.nan))
                V[name+ext] = np.full(tsi_max, np.nan)
        
        
        
        # initialize np arrays
        
        V['times'] = np.linspace(0, self.metadata['time_max'], tsi_max)
        V['position_x'] = np.full(tsi_max, np.nan)
        V['position_y'] = np.full(tsi_max, np.nan)
        V['position_z'] = np.full(tsi_max, np.nan)
        V['inPlume'] = np.full(tsi_max, -1, dtype=np.uint8)
        V['plume_experience'] = [None]*tsi_max
        V['turning'] = [None]*tsi_max
        V['heading_angle'] = np.full(tsi_max, np.nan)
        V['velocity_angular'] = np.full(tsi_max, np.nan)

        ######## INITIAL CONDITIONS
        # place agent
        V['position_x'][0], V['position_y'][0], V['position_z'][0] = place_agent(self.metadata['initial_position'])

        # generate random intial velocity condition using normal distribution fitted to experimental data
        V['velocity_x'][0], V['velocity_y'][0], V['velocity_z'][0] = np.random.normal(0, self.metadata['initial_velo_stdev'], 3)

        # generate a flight!
        for tsi in xrange(tsi_max):
            # import pdb; pdb.set_trace()
            # sense the temperature
#            _temperature[tsi] = self.plume_obj.temp_lookup([_position_x[tsi], _position_y[tsi]]) # depreciated

            # are we in the plume?
            if self.plume_obj.condition is None:  # skip if no plume
                V['inPlume'][tsi] = 0
            else:
                V['inPlume'][tsi] = self.plume_obj.check_for_plume(
                    [V['position_x'][tsi], V['position_y'][tsi], V['position_z'][tsi]])

            ############ what is our behavior given our plume interactions thus far?
            if tsi == 0:
                V['plume_experience'][tsi] = 'searching'
            else:
                if self.plume_obj.condition is None:  # hack for no plume condition
                    V['plume_experience'][tsi] = 'searching'
                else:
                    if V['plume_experience'][tsi] is None: # need to find state
                        V['plume_experience'][tsi] = self._check_crossing_state(tsi, V['inPlume'], V['velocity_y'][tsi-1])
                        # import pdb; pdb.set_trace()
                        if V['plume_experience'][tsi] in (
                                'Left_plume Exit leftLeft_plume Exit rightRight_plume Exit leftRight_plume Exit right'):
                            try:
                                for i in range(PLUME_TRIGGER_TIME): # store experience for the following timeperiod
                                    V['plume_experience'][tsi+i] = str(V['plume_experience'][tsi])
                            except IndexError: # can't store whole snapshot, so save 'truncated' label instead
                               # print "plume trigger turn lasted less than threshold of {} timesteps "\
                               # "before trajectory ended, so adding _truncated suffix".format(tsi_max)
                               for i in range(tsi_max - tsi):
                                    V['plume_experience'][tsi+i] = (str(V['plume_experience'][tsi])+'_truncated')

                    else: # if we're already orienting from past memory
                        if V['inPlume'] is False:  # we haven't re-entered the plume
                            pass
                        else:  # found the plume again!
                            V['plume_experience'][tsi:]
            ######################################################################


            # calculate driving forces' direction
            F_stim = stim_biasF3D.stimF(V['plume_experience'][tsi], self.metadata['stimF_strength'])
            V['stimF_x'][tsi], V['stimF_y'][tsi], V['stimF_z'][tsi] = F_stim

            F_bias = baseline_driving_forces3D.random_force(self.metadata['randomF_strength'])
            V['biasF_x'][tsi], V['biasF_y'][tsi], V['biasF_z'][tsi] = F_bias

            F_upwind = baseline_driving_forces3D.upwindBiasForce(self.metadata['wtF'])
            V['upwindF_x'][tsi], V['upwindF_y'][tsi], V['upwindF_z'][tsi] = F_upwind

            F_wall_repulsion = attraction_basin(self.metadata['k'], [V['position_x'][tsi], V['position_y'][tsi], V['position_z'][tsi]])
            V['wallRepulsiveF_x'][tsi], V['wallRepulsiveF_y'][tsi], V['wallRepulsiveF_z'][tsi] = F_wall_repulsion


            ########################### calculate current force
            F_driving =  F_bias + F_upwind + F_wall_repulsion + F_stim
            F_damping = -self.metadata['beta']*np.array([V['velocity_x'][tsi], V['velocity_y'][tsi], V['velocity_z'][tsi]])
            F_total = F_damping + F_driving
            V['totalF_x'][tsi], V['totalF_y'][tsi], V['totalF_z'][tsi] = F_total
            ###############################

            # calculate current acceleration
            V['acceleration_x'][tsi], V['acceleration_y'][tsi], V['acceleration_z'][tsi] = F_total / m

            # if time is out, end loop before we solve for future velo, position
            if tsi == tsi_max-1: # -1 because of how range() works
#                self.metadata['target_found'][0]  = False
#                self.metadata['time_to_target_find'][0] = np.nan
                V = self.land(tsi, V)
                break

            # calculate future velocity given the forces we just got
            V['velocity_x'][tsi+1], V['velocity_y'][tsi+1], V['velocity_z'][tsi+1] = \
                np.array([V['velocity_x'][tsi], V['velocity_y'][tsi], V['velocity_z'][tsi]]) \
                + np.array([V['acceleration_x'][tsi], V['acceleration_y'][tsi], V['acceleration_z'][tsi]]) * dt


            # solve candidate position for next timestep
            candidate_pos = np.array([V['position_x'][tsi], V['position_y'][tsi], V['position_z'][tsi]]) \
                + np.array([V['velocity_x'][tsi], V['velocity_y'][tsi], V['velocity_z'][tsi]])*dt # shouldnt position in future be affected by the forces in this timestep? aka use velo[i+1]
#
            # if walls are enabled, forbid mosquito from going out of bounds
            if bounded is True:
                # x dim
                if candidate_pos[0] > boundary[1]:  # reached far (upwind) wall (end)
#                    self.metadata['target_found'][0]  = False
#                    self.metadata['time_to_target_find'][0] = np.nan
                    V = self.land(tsi-1, V)  # stop flying at end, throw out last row
                    break
                if candidate_pos[0] < boundary[0]:  # too far left
#                    print "too far left"
                    candidate_pos[0] = boundary[0] + 1e-4 # teleport back inside
                    if BOUNCE == 'elastic':
                        V['velocity_x'][tsi+1] *= -1
#                        print "boom! left"
                    elif BOUNCE == 'crash':
                        V['velocity_x'][tsi+1] = 0.

                #y dim
                if candidate_pos[1] > boundary[2]:  # too left
#                    print "too left"
                    candidate_pos[1] = boundary[2] + 1e-4 # note, left is going more negative in our convention
                    if BOUNCE == 'elastic':
                        V['velocity_y'][tsi+1] *= -1
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V['velocity_y'][tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[1] < boundary[3]:  # too far right
#                    print "too far right"
                    candidate_pos[1] = boundary[3] - 1e-4
                    if BOUNCE == 'elastic':
                        V['velocity_y'][tsi+1] *= -1
                    elif BOUNCE == 'crash':
                        V['velocity_y'][tsi+1] = 0.

                # z dim
                if candidate_pos[2] > boundary[5]:  # too far above
#                    print "too far above"
                    candidate_pos[2] = boundary[5] - 1e-4
                    if BOUNCE == 'elastic':
                        V['velocity_z'][tsi+1] *= -1
#                        print "boom! top"
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V['velocity_z'][tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[2] < boundary[4]:  # too far below
#                    print "too far below"
                    candidate_pos[2] = boundary[4] + 1e-4
                    if BOUNCE == 'elastic':
                        V['velocity_z'][tsi+1] *= -1
                    elif BOUNCE == 'crash':
                        V['velocity_z'][tsi+1] = 0.

                # save screened candidate_pos to future position
            V['position_x'][tsi+1], V['position_y'][tsi+1], V['position_z'][tsi+1] = candidate_pos

            # solve final heading #TODO: also solve zy?
            V['heading_angle'][tsi] = solve_heading(V['velocity_y'][tsi], V['velocity_x'][tsi])
            # ... and angular velo
            if tsi == 0: # hack to prevent index error
                V['velocity_angular'][tsi] = 0
            else:
                V['velocity_angular'][tsi] = (V['heading_angle'][tsi] - V['heading_angle'][tsi-1]) / dt

            # turning state
            if tsi in [0, 1]:
                V['turning'][tsi] = 0
            else:
                turn_min = 3
                if abs(V['velocity_angular'][tsi-turn_min:tsi]).sum() > self.metadata['turn_threshold']*turn_min:
                    V['turning'][tsi] = 1
                else:
                    V['turning'][tsi] = 0

        return V


    def land(self, tsi, array_dict):
        ''' trim excess timebins in arrays
        '''
        for array in array_dict.itervalues():
            array = array[:tsi+1]

        array_dict = self._calc_polar_kinematics(array_dict)

        # absolute magnitude of velocity, accel vectors in 3D
        velo_mag_stack = np.vstack((array_dict['velocity_x'], array_dict['velocity_y'], array_dict['velocity_z']))
        array_dict['velocity_3Dmagn'] = np.linalg.norm(velo_mag_stack, axis=0)

        accel_mag_stack = np.vstack((array_dict['acceleration_x'], array_dict['acceleration_y'], array_dict['acceleration_z']))
        array_dict['acceleration_3Dmagn'] = np.linalg.norm(accel_mag_stack, axis=0)
        
        return array_dict
        
        
    def _check_crossing_state(self, tsi, inPlume, velocity_y):
        """
        out2out - searching
         (or orienting, but then this func shouldn't be called)
        out2in - entering plume
        in2in - staying
        in2out - exiting
            {Left_plume Exit left, Left_plume Exit right
#           Right_plume Exit left, Right_plume Exit right}
        """
        current_state, past_state = inPlume[tsi], inPlume[tsi-1]
        if current_state == 0 and past_state == 0:
            # we are not in plume and weren't in last ts
            return 'searching'
        if current_state == 1 and past_state == 0:
            # entering plume
            return 'entering'
        if current_state == 1 and past_state == 1:
            # we stayed in the plume
            return 'staying'
        if current_state == 0 and past_state == 1:
            # exiting the plume
            if self.metadata['heater_position'][5] == 'left':
                if velocity_y <= 0:
                    return 'Left_plume Exit left'
                else:
                    return 'Left_plume Exit right'
            else:
                if velocity_y <= 0:
                    return "Right_plume Exit left"
                else:
                    return "Right_plume Exit right"
            
            
    def _calc_polar_kinematics(self, array_dict):
        """append polar kinematics to vectors dictionary"""
        for name in self.metadata['kinematic_vals']+self.metadata['forces']:  # ['velocity', 'acceleration', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
            x_component, y_component = array_dict[name+'_x'], array_dict[name+'_y']
            angle = np.arctan2(y_component, x_component)
            angle[angle < 0] += 2*np.pi  # get vals b/w [0,2pi]
            array_dict[name+'_xy_theta'] = angle
            array_dict[name+'_xy_mag'] = np.sqrt(y_component**2 + x_component**2)
        
        return array_dict



def main():
    """
    Params fitted using scipy.optimize

    """

    N_TRAJECTORIES = 1
    TEST_CONDITION = None  # {'Left', 'Right', None}
    # old beta- 5e-5, forces 4.12405e-6, fwind = 5e-7
    BETA, FORCES_AMPLITUDE, F_WIND_SCALE =  [  1.37213380e-06  , 1.39026239e-06 ,  2.06854777e-08]
    MASS = 2.88e-6
    F_STIM_SCALE = 0.  #7e-7,   # set to zero to disable tracking hot air
    K = 0.  #1e-7               # set to zero to disable wall attraction

    # generate environment
    windtunnel_object = windtunnel.Windtunnel(TEST_CONDITION)
    # generate temperature plume
    plume_object = plume3D.Plume(TEST_CONDITION)
    # instantiate empty trajectories class
    trajectories_object = trajectory3D.Trajectory()
    # instantiate a Roboskeeter
    skeeter = Agent(
        trajectories_object,
        plume_object,
        windtunnel_object,
        mass=MASS,
        agent_pos="downwind_plane",
        windF_strength=F_WIND_SCALE,
        randomF_strength=FORCES_AMPLITUDE,
        stimF_str=F_STIM_SCALE,
        k=K,
        beta=BETA,
        Tmax=15.,
        dt=0.01,
        bounded=True)
    sys.stdout.write("\rAgent born")

    # make the skeeter fly. this updates the trajectories_object
    skeeter.fly(total_trajectories=N_TRAJECTORIES)
    
    return plume_object, skeeter.trajectory_obj, skeeter
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    #trajectories.plot_single_3Dtrajectory()
#    
    

if __name__ == '__main__':
    myplume, trajectories, skeeter = main()

    print "\nDone."

    ######################### plotting methods
    # trajectories.plot_single_3Dtrajectory(0)  # plot ith trajectory in the ensemble of trajectories

    # trajectories.plot_force_violin()
    # trajectories.plot_kinematic_hists()
    # trajectories.plot_posheatmap()
    # trajectories.plot_kinematic_compass()
    # trajectories.plot_sliced_hists()

    ######################### dump data for csv for Sharri
    # print "dumping to csvs"
    # e = trajectories.ensemble
    # r = e.trajectory_num.iloc[-1]
    #
    # for i in range(r+1):
    #     e1 = e.loc[e['trajectory_num'] == i, ['position_x', 'position_y', 'position_z', 'inPlume']]
    #     e1.to_csv("l"+str(i)+'.csv', index=False)
    #
    # # trajectories.plot_kinematic_hists()
    # # trajectories.plot_posheatmap()
    # # trajectories.plot_force_violin()
    #
    # # # for plume stats
    # # g = e.loc[e['plume_experience'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]