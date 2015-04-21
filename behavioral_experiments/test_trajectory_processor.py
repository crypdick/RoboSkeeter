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
import unittest


def NaN_run(len):
    data = df(np.full((len, 3), np.nan), columns = ['x','y','z'])
    return data
    
    
def num_run(len):
    data = df(np.random.randn(len, 3), columns = ['x','y','z'])
    return data


## Building blocks
NaN20 = NaN_run(20)
NaN50 = NaN_run(50)
NaN100 = NaN_run(100)

num20 = num_run(20)
num50 = num_run(50)
num100 = num_run(100)

## test trajectories

#run = num_run(100)
#print run
#print "------------------------------"
#run2 = flight_trajectory_processor.trim_NaNs(run)
#
def check_df_equal(df1, df2):
    from pandas.util.testing import assert_frame_equal
    try:
        assert_frame_equal(df1.sort(axis=1), df2.sort(axis=1), check_names=True)
        return True
    except (AssertionError, ValueError, TypeError):
        return False
#
#print my_equal(run, run2)


class TestStringMethods(unittest.TestCase):
    from pandas.util.testing import assert_frame_equal

    def test_trimNaN_nums(self): # only numbers
        df = num_run(100)
        df2 = flight_trajectory_processor.trim_NaNs(df)
        self.assertTrue(check_df_equal(df, df2))
        
        def test_trimNaN_nums(self): # only numbers
        df = num_run(100)
        df2 = flight_trajectory_processor.trim_NaNs(df)
        self.assertTrue(check_df_equal(df, df2))
        
    def test_allNums(self): # only numbers

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
    pass




# trajectory, all NaNs
# trajectory, some NaNs
# trajectory, some NaN stretches, len < 50
# trajectory, exactly 50 NaNs
# trajectory, some NaN stretches, len > 50
# trajectory, leading NaNs
# trajectory, trailing NaNs
# trajectory, 
