# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:34:50 2015

@author: richard
"""

#from __future__ import print_function
import glob
import os
import flight_trajectory_processor



os.chdir("trajectory_data/")
for file in glob.glob("*.csv"):
    abspath = os.path.abspath(file)
    print abspath
    flight_trajectory_processor.main(abspath)