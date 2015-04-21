# -*- coding: utf-8 -*-
"""
testing our processor

Created on Tue Apr 21 14:43:17 2015

@author: richard
"""
import pandas as pd
from pandas import concat as cat
from pandas import DataFrame as df
import numpy as np
import flight_trajectory_processor


def NaN_run(len):
    return df(np.full((len, 3), np.nan))
    
def num_run(len):
    return df(np.random.randn(len, 3))
## Building blocks

NaN20 = NaN_run(20)
NaN50 = NaN_run(50)
NaN100 = NaN_run(100)

num20 = num_run(20)
num50 = num_run(50)
num100 = num_run(100)

## test trajectories

# trajectory, all numbers
allNums = num_run(100)
output = flight_trajectory_processor.main(allNums)

# trajectory, all NaNs
# trajectory, some NaNs
# trajectory, some NaN stretches, len < 50
# trajectory, exactly 50 NaNs
# trajectory, some NaN stretches, len > 50
# trajectory, leading NaNs
# trajectory, trailing NaNs
# trajectory, 
