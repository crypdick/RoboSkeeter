__author__ = 'richard'

import trajectory

SELECTION = None  # 'CONTROL_EXP_PATH'

control = trajectory.Experimental_Trajectory()
control.load_experiments(selection=SELECTION)



################### helper lines
# # return sorted trajectory nums
# np.sort(control.data.trajectory_num.unique())


# control.plot_3Dtrajectory(13)  # jagged traj
#  control.plot_3Dtrajectory(14)  # super curvy
#
# control.plot_3Dtrajectory(34)  # short
# control.plot_3Dtrajectory(57)  # short
# control.plot_3Dtrajectory(44)  # short
# control.plot_3Dtrajectory(51)  # short
# control.plot_3Dtrajectory(139)  # short
# control.plot_3Dtrajectory(141)  # short


#
# ########### show all trajectories that make it to heater
# above_heater = control.data.loc[(control.data['position_x'] > 0.865)]
# traj_is = above_heater.trajectory_num.unique()
#
# for i in traj_is:
#     i = int(i)
#     control.plot_3Dtrajectory(i)
