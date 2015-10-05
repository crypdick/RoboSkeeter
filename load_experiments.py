__author__ = 'richard'

import trajectory
from scripts import i_o

control_dir = i_o.get_directory('CONTROL_EXP_PATH')
control = trajectory.Experimental_Trajectory()
control.load_experiments(directory=control_dir)
