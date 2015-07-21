# -*- coding: utf-8 -*-
"""
make our agents fly towards the plume.


"""
import numpy as np

# depreciated
#def absoluteT_threshold_check(temperature, threshold):#, strength, inplume_past):
#    """given temperature, check if it's above our threshold
#    return if we're in plume (bool)"""
#    if temperature >= threshold:
#        return True
#    else:
#        return False




def stimF(experience, stimF_strength):
    """given force direction and strength, return a force vector
    Args:
    experience: {searching, entering, Left_plume, exit left, Left_plume, exit right, Right_plume, exit left, Right_plume, exit right}
    """
    if experience in 'searchingentering':
        return np.array([0., 0., 0.])
    elif experience in 'staying':
        return np.array([stimF_strength, 0., 0.])  # surge while in plume
    elif "Exit left" in experience:
        return np.array([0., stimF_strength, 0.])
    elif "Exit right" in experience:
        return np.array([0., -stimF_strength, 0.])
    else:
        print "error", experience
        return np.array([0., 0., 0.])

#([0., self.stimF_str, 0.]),
   
#def main velocity_y, stimF_strength, past_state=False, detection_type='absolute'):
#    veloy = None
#    if detection_type == 'absolute':
#        curr_status = absoluteT_threshold_check(temperature, thresh)
#    cross_state = check_crossing_state(curr_status, past_state)
#    stimF_dir = stimF_direction(cross_state, veloy)
#    stim_F = stimF(stimF_strength, stimF_dir)
#    
#    return stim_F, curr_status
     
#==============================================================================
# # depreciated
# #def main(temperature, velocity_y, stimF_strength, thresh=299.15, past_state=False, detection_type='absolute'):
# #    veloy = None
# #    if detection_type == 'absolute':
# #        curr_status = absoluteT_threshold_check(temperature, thresh)
# #    cross_state = check_crossing_state(curr_status, past_state)
# #    stimF_dir = stimF_direction(cross_state, veloy)
# #    stim_F = stimF(stimF_strength, stimF_dir)
     #    
#    return stim_F, curr_status
#==============================================================================

    
    
    