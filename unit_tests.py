__author__ = 'richard'

import unittest
import numpy as np
from pandas import concat
from pandas import DataFrame as df

from scripts import process_flight_data

class TestSorcery(unittest.TestCase):
    def __init__(self):
        pass

    # def setUp(self):
    #     pass
    #
    # def test_curvature(self):
    #     self.assertEqual( calculate_curvature(3,4), 12)
    #
    # def test_heading_vecs(self):
    #     self.assertEqual( multiply('a',3), 'aaa')


class TestProcessor(unittest.TestCase):
    def __init__(self):
        ## Building blocks
        self.NaN20 = self.NaN_run(30)
        self.NaN50 = self.NaN_run(50)
        self.NaN100 = self.NaN_run(100)

        self.num10 = self._num_run(10)
        self.num20 = self._num_run(20)
        self.num21 = self._num_run(21)
        self.num30 = self._num_run(30)
        self.num50 = self._num_run(50)
        self.num100 = self._num_run(100)

    def _NaN_run(self, len):
        data = df(np.full((len, 3), np.nan), columns = ['x','y','z'])
        return data

    def _num_run(self, len):
        data = df(np.random.randn(len, 3), columns = ['x','y','z'])
        return data

    def check_df_equal(self, df1, df2):
        from pandas.util.testing import assert_frame_equal
        try:
            assert_frame_equal(df1.sort(axis=1), df2.sort(axis=1), check_names=True)
            return True
        except (AssertionError, ValueError, TypeError):
            return False

    ## TEST TRIMMER
#    def test_trimNaN_nums(self): # only numbers
#        df1 = num_run(100)
#        df2 = flight_trajectory_processor.trim_NaNs(df1)
#        self.assertTrue(self.check_df_equal(df1, df2))

#    def test_trimNaN_nums(self): # only NaNs
#        df = NaN_run(100)
#        df2 = flight_trajectory_processor.trim_NaNs(df)
#        self.assertTrue(self.check_df_equal(df([]), df2)) # todo: figure out how to make empty dataframe

    ## TEST SPLITTER
    def test_allNums(self): # only numbers
        df1 = self.self.num100
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal(df1, df2[0]))

    def test_someNaNs(self):
        df1 = concat([self.num50, self.NaN20, self.num30]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal(df1, df2[0]))

    def test_manyNaNs(self):
        df1 = concat([self.num50, self.NaN100, self.num30]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal( self.num50,  df2[0]))
        self.assertTrue(self.check_df_equal( self.num30,  df2[1]))


    def test_leadingNaNs(self):
        df1 = concat([self.NaN100, self.num50, self.NaN100, self.num30, self.NaN100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal( self.num50,  df2[0]))
        self.assertTrue(self.check_df_equal( self.num30,  df2[1]))


    def test_trailingNaNs(self):
        df1 = concat([self.num50, self.NaN100, self.num30, self.NaN100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal( self.num50,  df2[0]))
        self.assertTrue(self.check_df_equal( self.num30,  df2[1]))

    def test_short_trajectory(self):
        df1 = concat([self.num50, self.NaN100, self.num10, self.NaN100, self.self.num100]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal( self.num50,  df2[0]))
        self.assertTrue(self.check_df_equal( self.self.num100,  df2[1]))

    def test_50NaNs(self):
        df1 = concat([self.num21, self.NaN50, self.num21]).reset_index()[['x', 'y', 'z']]
        df2 = process_flight_data.main(df1)
        self.assertTrue(self.check_df_equal(self.num21, df2[0]))
        self.assertTrue(self.check_df_equal(self.num21, df2[1]))