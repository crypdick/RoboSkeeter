__author__ = 'richard'

import agent

N_TRAJECTORIES = 5
TEST_CONDITION = None  # {'Left', 'Right', None}
# old beta- 5e-5, forces 4.12405e-6, fwind = 5e-7
BETA, RANDF_STRENGTH, F_WIND_SCALE = [1.37213380e-06, 1.39026239e-06, 7.06854777e-07]
F_STIM_SCALE = 0.  # 7e-7,   # set to zero to disable tracking hot air
MASS = 2.88e-6
K = 0.  # 1e-7               # set to zero to disable wall attraction
BOUNDED = False

trajectories, skeeter = agent.gen_objects_and_fly(
    N_TRAJECTORIES,
    TEST_CONDITION,
    BETA,
    RANDF_STRENGTH,
    F_WIND_SCALE,
    F_STIM_SCALE,
    MASS,
    K,
    bounded=BOUNDED
)

print "\nDone."

######################### plotting methods
trajectories.plot_single_3Dtrajectory(0)  # plot ith trajectory in the ensemble of trajectories

# trajectories.plot_kinematic_hists()
# trajectories.plot_posheatmap()
# trajectories.plot_force_violin()  # TODO: fix Nans in arrays
# trajectories.plot_kinematic_compass()
# trajectories.plot_sliced_hists()

######################### dump data for csv for Sharri
# print "dumping to csvs"
# e = trajectories.data
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
# # g = e.loc[e['behavior_state'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]
