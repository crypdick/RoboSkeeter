__author__ = 'richard'
import numpy as np
from math import atan2

def calculate_heading(velo_x_component, velo_y_component):
    theta = atan2(velo_y_component, velo_x_component)

    return theta*180/np.pi


def calculate_curvature(ensemble):
    velo_vector = np.vstack((ensemble.velocity_x, ensemble.velocity_y, ensemble.velocity_z)  # shape is (3, R)
    accel_vector = np.vstack((ensemble.acceleration_x, ensemble.acceleration_y, ensemble.acceleration_z))
    # using formula from https://en.wikipedia.org/wiki/Curvature#Local_expressions_2
     return np.abs( np.cross(velo_vector, accel_vector) ) /                           \
            np.linalg.norm(accel_vector, axis=0)** 3


def gen_symm_vecs(dims):
    """sample random (radially-symmetric) 3D vectors from the unit sphere.

    credit: http://codereview.stackexchange.com/a/77945/76407
    """
    vecs = np.random.normal(size=dims)
    mags = np.linalg.norm(vecs, axis=-1)

    ends = vecs / mags[..., np.newaxis]  # divide by length to get unit vector

    return ends