# -*- coding: utf-8 -*-
"""
testing our processor

Created on Tue Apr 21 14:43:17 2015

@author: richard
"""
import numpy as np
import unittest

from pandas import concat
from pandas import DataFrame as df

import process_flight_data


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

num10 = num_run(10)
num20 = num_run(20)
num21 = num_run(21)
num30 = num_run(30)
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
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal(df1, df2[0]))
        
    def test_someNaNs(self):
        df1 = concat([num50, NaN20, num30]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal(df1, df2[0]))
    
    def test_manyNaNs(self):
        df1 = concat([num50, NaN100, num30]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal( num50,  df2[0]))
        self.assertTrue(check_df_equal( num30,  df2[1]))
        
        
    def test_leadingNaNs(self):
        df1 = concat([NaN100, num50, NaN100, num30, NaN100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal( num50,  df2[0]))
        self.assertTrue(check_df_equal( num30,  df2[1]))
        
        
    def test_trailingNaNs(self):
        df1 = concat([num50, NaN100, num30, NaN100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal( num50,  df2[0]))
        self.assertTrue(check_df_equal( num30,  df2[1]))
        
    def test_short_trajectory(self):
        df1 = concat([num50, NaN100, num10, NaN100, num100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal( num50,  df2[0]))
        self.assertTrue(check_df_equal( num100,  df2[1]))
        
    def test_50NaNs(self):
        df1 = concat([num21, NaN50, num21]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(check_df_equal(num21, df2[0]))
        self.assertTrue(check_df_equal(num21, df2[1]))


if __name__ == '__main__':
    unittest.main()


# trajectory, all NaNs
# trajectory, some NaN stretches, len < 50
# trajectory, exactly 50 NaNs

