mozziesniff
===========

Modeling mosquito decision making in turbulent, thermal plumes.

generate_trajectory generates a single trajectory and can take inputs.
trajectory_stats runs generate_trajectory many times and uses the plotting_funcs script to visualize the stats.
Pfind_stats divides the "wind tunnel" into a grid and iterates through each cell and places the target in that cell. Then, it runs a bunch of trajectories and scores the probability of our agent entering into that cell.

 
Mozzie_plummer, mozzie-timestepper are the older agent model controlled with 2 leaky integrate-and-fire neurons. Ginglsensor uses data from Gingl's paper to determine neuron fire rates given absolute temp and change in temp.