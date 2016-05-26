"""
This is a script I used to figure out what binwidth I should use for my kernel density estimates for each kinematic distribution

I used the code from Jake VDP's site here: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

For each independent kinematic (position_x _y _z, acceleration, velocity, and curvature) I did a 20-fold cross validation
"""
from roboskeeter import experiments
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
import time

n_cross_validations = 2  # minimum 2

# the kinematics we want to score
# kinematics_list = ['velocity_x', 'velocity_y', 'velocity_z', 'curvature', 'position_x', 'position_y', 'position_z', 'acceleration_x', 'acceleration_y', 'acceleration_z']
kinematics_list = ['curvature', 'velocity_y']

control_conditions = {'condition': 'Control',  # {'Left', 'Right', 'Control'}
                         'plume_model': "None",  # "Boolean" "None, "Timeavg", "Unaveraged"
                         'time_max': "N/A (experiment)",
                         'bounded': True,
                         }

L_conditions = {'condition': 'Left',  # {'Left', 'Right', 'Control'}
                         'plume_model': "None",  # "Boolean" "None, "Timeavg", "Unaveraged"
                         'time_max': "N/A (experiment)",
                         'bounded': True,
                         }

R_conditions = {'condition': 'Right',  # {'Left', 'Right', 'Control'}
                         'plume_model': "None",  # "Boolean" "None, "Timeavg", "Unaveraged"
                         'time_max': "N/A (experiment)",
                         'bounded': True,
                         }

conditions = [control_conditions, L_conditions, R_conditions]
# conditions = [control_conditions]

for condition in conditions:
    experiment = experiments.load_experiment(condition)
    kin = experiment.observations.kinematics
    kinematics_dict = {'velocity_x': kin.velocity_x.values,
                       'velocity_y': kin.velocity_y.values,
                       'velocity_z': kin.velocity_y.values,
                       'curvature': kin.curvature.values,
                       'acceleration_x': kin.acceleration_x.values,
                       'acceleration_y': kin.acceleration_y.values,
                       'acceleration_z': kin.acceleration_z.values}
    for kinematic in kinematics_list:
        x = kinematics_dict[kinematic]

        print "gridsearch started for {} - condition = {}".format(kinematic, condition['condition'])
        start = time.clock()
        grid = GridSearchCV(KernelDensity(),
                            # {'bandwidth': np.linspace(0.02, .2, 10)},
                            {'bandwidth': np.linspace(0.01, .1, 4)},
                            cv=n_cross_validations)
        grid.fit(x[:, None])
        end = time.clock()
        print "{}-fold crossvalidation gridsearch for {} finished after {}".format(n_cross_validations, kinematic, (end-start))
        print grid.best_params_



# x_grid = np.linspace(-5, 5, 50)
# kde_skl = KernelDensity(bandwidth=0.02)
# kde_skl.fit(x[:, np.newaxis])
# log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
# pdf = np.exp(log_pdf)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
# plt.show()