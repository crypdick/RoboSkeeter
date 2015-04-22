# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:34:50 2015

@author: richard
"""

from __future__ import print_function
import glob
import os
os.chdir("/mydir")
for file in glob.glob("*.txt"):
    print(file)