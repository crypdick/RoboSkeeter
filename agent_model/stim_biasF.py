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
"""
import numpy as np

#def force(dir=None):
#    if dir=None:
#        return [0, 0]

def abs_plume(position, strength, inplume_past):
    """given postion, return if we're in plume (bool), and what the forces
    should be"""
    dummy = True
    if dummy is True:
        return np.array([0, 0]), False
    else: #
        # what is the temp at this position?
        return np.array([0, 0]), False