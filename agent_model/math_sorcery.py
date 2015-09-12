__author__ = 'richard'
import numpy as np
from math import atan2

def calculate_heading(velo_x_component, velo_y_component):
    theta = atan2(velo_y_component, velo_x_component)

    return theta*180/np.pi


def calculate_curvature(ensemble):
    """using formula from https://en.wikipedia.org/wiki/Curvature#Local_expressions_2"""
    velo_vector = np.vstack((ensemble.velocity_x, ensemble.velocity_y, ensemble.velocity_z)  # shape is (3, R)
    accel_vector = np.vstack((ensemble.acceleration_x, ensemble.acceleration_y, ensemble.acceleration_z))
     return np.abs( np.cross(velo_vector, accel_vector) ) /                           \
            np.linalg.norm(accel_vector, axis=0)** 3


def gen_symm_vecs(dims=3):
    """generate randomly pointed (radially-symmetric) 3D unit vectors/ direction vectors

    first we draw from a 3D gaussian, which is a symmetric distribution no matter how you slice it. then, we map
    those draws onto the unit sphere.

    credit: http://codereview.stackexchange.com/a/77945/76407
    """
    vecs = np.random.normal(size=dims)
    mags = np.linalg.norm(vecs, axis=-1)

    ends = vecs / mags[..., np.newaxis]  # divide by length to get unit vector

    return ends