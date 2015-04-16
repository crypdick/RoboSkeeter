# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 07:59:47 2015

@author: Richard Decal

bootstrapping on upwind-left vs -right quadrants.

To compare upwind positional distributions, we examined the percentage of trajectory positions found in the left half of the wind tunnel. For each experimental condition, we can calculate this left side bias exactly once. We use this bias to determine if the experimental upwind distributions of position are likely to be drawn from the upwind control distribution of position. We used bootstrapping methods to collect samples from the control distribution of position, and these samples are repeatedly drawn in order to find the probability distribution of the left side bias values.

Position Samples -  To build our sample set, positions from the control trajectories were randomly selected. We built our samples such that the number of points drawn equal the number of points seen in either of the experimental conditions. Samples of positions were collected in two ways: individual points and whole trajectories.

Individual points were selected randomly from any trajectory. This sampling strategy allows us to compare probabilities in a truly random fashion, however, position values found along a mosquito trajectory are highly correlated, and thus are not truly random. To ensure that the correlation seen along a trajectory does not influence the bias or distribution seen, we randomly selected entire trajectories.

The point or trajectory samples were collected in order to get a single side bias value. However, to determine the probability of seeing the side bias by chance, one needs to know the distribution of side bias values. Thus, the samples were redrawn and the left side biases were calculated 10 000 times. The distribution was well-fit by a Guassian, which, by assuming the observed mean and standard deviation are close or equal to the expected mean and standard deviation, can be used to calculate statistical significance of the observed left side bias. Here, solving for the 95% confidence interval of the distribution of left side biases equivocates to the probability of drawing a given left side bias value from chance. That is, if the left side biases calculated from the experiments are greater than twice the standard deviation of the distribution, the p-value is less than 0.05.

"""

from Agent
