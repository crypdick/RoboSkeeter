# -*- coding: utf-8 -*-
"""
make our agents fly towards the plume.

in_Plume = True/False state-abs temp thresh
if in_Plume switches from True->False,
	check sign of vy and take the opposite (head back to plume)
	F = Bv +rf + wtf + repulsion_landscape + plume_bias(direction, strength, _decay_)
	direction = {left, right, None}	
	loose casting behavior
check how this biases heatmap

Created on Thu Apr 23 13:04:37 2015

@author: richard

in_Plume = True/False state-abs temp thresh
if in_Plume switches from True->False,
	check sign of vy and take the opposite (head back to plume)
         if vy > 0...
	F = Bv +rf + wtf + repulsion_landscape + plume_bias(direction, strength, _decay_)
	direction = {left, right, None}	
	loose casting behavior
check how this biases heatmap

STEP 2
whene xit plume again, come back.

plume policy:
inPlume = True/False
mozzie.temp() ==>  plume class
plumeDirection = [left, right, none]
= .... + stimBias(plumeDir, strength)


"""

def absoluteT_threshold_check(temperature, threshold):#, strength, inplume_past):
    """given temperature, check if it's above our threshold
    return if we're in plume (bool)"""
    if temperature >= threshold:
        return True
    else:
        return False
        

def check_crossing_state(current_state, past_state):
    """in2in, out2in - stay, entering plume
    out2out - off, searching
    in2out - reverse, exiting
    """
    if current_state == False and past_state == False:
        # we are not in plume and weren't in last ts
        return 'searching'
    if current_state == True and past_state == False:
        # entering plume
        return 'entering'
    if current_state == True and past_state == True:
        # we stayed in the plume
        return 'staying'
    if current_state == False and past_state == True:
        # exiting the plume
        return "exiting"


def stimF_direction(cross_status, veloy):
    """Given the crossing state and current crosswind velocity, return which
    direction we should be flying in.
    
    TODO: solve for direction lookkup given cross status (searching, entering, staying, exiting)
    """
    return None


def stimF(stimF_strength, dir=None):
    """given force direction and strength, return a force vector
    
    TODO: add forces for other directions
    """
    if dir == None:
        return [0, 0, 0]
        
        
def main(temperature, velocity_y, stimF_strength, thresh=299.15, past_state=False, detection_type='absolute'):
    veloy = None
    if detection_type == 'absolute':
        curr_status = absoluteT_threshold_check(temperature, thresh)
    cross_state = check_crossing_state(curr_status, past_state)
    stimF_dir = stimF_direction(cross_state, veloy)
    stim_F = stimF(stimF_strength, stimF_dir)
    
    return stim_F, curr_status
    
    
    