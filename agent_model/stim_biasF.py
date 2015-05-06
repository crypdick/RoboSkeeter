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
import numpy as np


def absoluteT_detect(temperature, threshold = 299.15):#, strength, inplume_past):
    """given temperature, check if it's above our threshold
    return if we're in plume (bool)"""
    if temperature >= threshold:
        return True
    else:
        return False
        

def bound_crossing_status(current_status, past_status):
    """in2in, out2in - stay, entering plume
    out2out - off, searching
    in2out - reverse, exiting
    """
    if current_status == False and past_status == False:
        # we are not in plume and weren't in last ts
        return 'searching'
    if current_status == True and past_status == False:
        # entering plume, or staying in plume
        return 'entering'
    if current_status == False and past_status == True:
        # exiting the plume
        return "exiting"


def stimF_direction(cross_status, veloy):
    # TODO
    return None


def stimF(dir=None):
    if dir == None:
        return [0, 0]
        
        
def main(temperature, past_status, detection_type='absolute'):
    veloy = None
    if detection_type == 'absolute':
        curr_status = absoluteT_detect(temperature, thresh)
    cross_status = bound_crossing_status(curr_status, past_status)
    stimF_dir = stimF_direction(cross_status, veloy)
    force = stimF()
    
    return curr_status, force
    
    
    