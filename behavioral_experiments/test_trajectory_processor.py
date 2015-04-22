# -*- coding: utf-8 -*-
"""
testing our processor

Created on Tue Apr 21 14:43:17 2015

@author: richard
"""
import pandas as pd
from pandas import concat
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
NaN20 = NaN_run(30)
NaN50 = NaN_run(50)
NaN100 = NaN_run(100)

num20 = num_run(30)
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

    ## TEST TRIMMER
#    def test_trimNaN_nums(self): # only numbers
#        df1 = num_run(100)
#        df2 = flight_trajectory_processor.trim_NaNs(df1)
#        self.assertTrue(check_df_equal(df1, df2))
        
#    def test_trimNaN_nums(self): # only NaNs
#        df = NaN_run(100)
#        df2 = flight_trajectory_processor.trim_NaNs(df)
#        self.assertTrue(check_df_equal(df([]), df2)) # todo: figure out how to make empty dataframe
    
    ## TEST SPLITTER
    def test_allNums(self): # only numbers
        df1 = num100
        df2 = flight_trajectory_processor.main(df1)
        self.assertTrue(check_df_equal(df1, df2[0]))
        
    def test_someNaNs(self):
        df1 = concat([num50, NaN20, num20]).reset_index()[['x', 'y', 'z']]
        df2 = flight_trajectory_processor.main(df1)
        self.assertTrue(check_df_equal(df1, df2[0]))
    
    def test_manyNaNs(self):
        df1 = concat([num50, NaN100, num20]).reset_index()[['x', 'y', 'z']]
        df2 = flight_trajectory_processor.main(df1)
        self.assertTrue(check_df_equal( num50.reset_index()[['x', 'y', 'z']],  df2[0]))
        self.assertTrue(check_df_equal( num20.reset_index()[['x', 'y', 'z']],  df2[1]))
        
#
#    def test_split(self):
#        s = 'hello world'
#        self.assertEqual(s.split(), ['hello', 'world'])
#        # check that s.split fails when the separator is not a string
#        with self.assertRaises(TypeError):
#            s.split(2)

if __name__ == '__main__':
    unittest.main()
    pass




# trajectory, all NaNs
# trajectory, some NaN stretches, len < 50
# trajectory, exactly 50 NaNs
# trajectory, leading NaNs
# trajectory, trailing NaNs
# trajectory, 
