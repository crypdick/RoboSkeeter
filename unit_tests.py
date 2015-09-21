__author__ = 'richard'

import unittest


class TestSorcery(unittest.TestCase):

    def setUp(self):
        pass

    def test_curvature(self):
        self.assertEqual( calculate_curvature(3,4), 12)

    def test_heading_vecs(self):
        self.assertEqual( multiply('a',3), 'aaa')


if __name__ == '__main__':
    unittest.main()