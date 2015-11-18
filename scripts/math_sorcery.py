from __future__ import print_function, division

import numpy as np
from sklearn.neighbors import KernelDensity

__author__ = 'richard'


def norm(vectors, shape='1darray'):
    """
    Calculate the norm, making sure to return a float.
    Credit: rkp8000/windtunnel

    :param vectors: 2D array of vectors
    :param shape: whether to return '1darray' or '2darray' w/ same shape as vectors
    :return: vector norm
    """

    n = np.linalg.norm(vectors.astype(float), axis=1)

    if shape == '1darray':
        return n
    elif shape == '2darray':
        return np.tile(n, (vectors.shape[1], 1)).T


def angular_velocity(velocities, dt):
    """
    Calculate angular velocities.
    Credit: rkp8000/wind_tunnel

    :param velocities: 2D array of velocities (rows are timepoints)
    :param dt: interval between timesteps
    :return: array of angular velocities
    """

    # calculate normalized velocity vector
    v_norm = velocities / norm(velocities, shape='2darray')

    # get angle between each consecutive pair of normalized velocity vectors
    d_theta = np.arccos((v_norm[:-1, :] * v_norm[1:, :]).sum(1))
    a_vel_mag = d_theta / dt
    # calculate the direction of angular change by computing the cross-
    # product between each consecutive pair of normalized velocity vectors
    cp = np.cross(v_norm[:-1, :], v_norm[1:, :])
    # normalize the cross product array
    cp /= np.tile(np.linalg.norm(cp.astype(float), axis=1), (3, 1)).T
    # create angular velocity array and set to zero the places where the magnitude is zero
    a_vel = cp * np.tile(a_vel_mag, (3, 1)).T
    a_vel[a_vel_mag == 0] = 0
    # correct size so that it matches the size of the velocity array
    a_vel_full = np.zeros((a_vel.shape[0] + 1, a_vel.shape[1]), dtype=float)
    a_vel_full[:-1] += a_vel
    a_vel_full[1:] += a_vel
    a_vel_full[1:-1] /= 2.

    return a_vel_full


def heading(velocities):
    """
    Calculate heading in xy and xz plane, as well as 3d heading.
    Credit: rkp8000/windtunnel

    :param velocities: 2D array of velocities (rows are timepoints)
    :return: array of headings (first col xy, second col xz, third col xyz)
    """

    v_xy = velocities[:, [0, 1]].copy().astype(float)
    v_xz = velocities[:, [0, 2]].copy().astype(float)
    v_xyz = velocities.copy().astype(float)

    # normalize each set of velocities
    norm_xy = norm(v_xy)
    norm_xz = norm(v_xz)
    norm_xyz = norm(v_xyz)

    v_xy /= np.tile(norm_xy, (2, 1)).T
    v_xz /= np.tile(norm_xz, (2, 1)).T
    v_xyz /= np.tile(norm_xyz, (3, 1)).T

    # array of upwind vectors
    uw_vec = np.transpose([-np.ones((len(velocities),), dtype=float),
                           np.zeros((len(velocities),), dtype=float),
                           np.zeros((len(velocities),), dtype=float)])

    heading_xy = np.arccos((uw_vec[:, [0, 1]] * v_xy).sum(axis=1))
    heading_xz = np.arccos((uw_vec[:, [0, 2]] * v_xz).sum(axis=1))
    heading_xyz = np.arccos((uw_vec * v_xyz).sum(axis=1))

    heading_xy[norm_xy == 0] = 0
    heading_xz[norm_xz == 0] = 0
    heading_xyz[norm_xyz == 0] = 0

    return np.transpose([heading_xy, heading_xz, heading_xyz]) * 180 / np.pi

def calculate_curvature(ensemble):
    """using formula from https://en.wikipedia.org/wiki/Curvature#Local_expressions_2"""
    velocity_vec = np.vstack((ensemble.velocity_x, ensemble.velocity_y, ensemble.velocity_z))  # shape is (3, R)
    acceleration_vec = np.vstack((ensemble.acceleration_x, ensemble.acceleration_y, ensemble.acceleration_z))

    numerator = np.linalg.norm( np.cross(velocity_vec.T, acceleration_vec.T), axis=1)
    denominator = np.linalg.norm(acceleration_vec, axis=0)** 3

    curvature = numerator/denominator

    curvature = np.nan_to_num(curvature)  # hack to prevent curvature nans

    return curvature


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

def rads_to_degrees(rads):
    degrees = (rads * 180/np.pi) % 360  # map to [0,360)
    return degrees


def calculate_xy_heading_angle(x_component, y_component):
    angle = np.arctan2(y_component, x_component)
    angle = rads_to_degrees(angle)
    return angle

def calculate_xy_magnitude(x_component, y_component):
    return np.sqrt(x_component**2 + y_component**2)


def is_turning():
    pass
"""            # # turning state
            # if tsi in [0, 1]:
            #     V['turning'][tsi] = 0
            # else:
            #     turn_min = 3
            #     if abs(V['velocity_angular'][tsi-turn_min:tsi]).sum() > self.turn_threshold*turn_min:
            #         V['turning'][tsi] = 1
            #     else:
            #         V['turning'][tsi] = 0"""


def calculate_1Dkde(vector, bandwidth=0.5):
    """

    :param vector: a 1D vector
    :return: kde object
    """
    kernel = 'gaussian'
    vector = vector[:, np.newaxis]

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(vector)

    return kde


def evaluate_kde(kde, bins):
    log_dens = kde.score_samples(bins.reshape(-1, 1))  # fixes error: http://stackoverflow.com/a/27748566
    dens = np.exp(log_dens)

    return dens


def distance_from_wall(positions, wall_bounds):
    """
    Calculate distance from nearest wall. Credit: rkp8000/wind_tunnel

    :param positions: 2D array of positions (rows are timepoints)
    :param wall_bounds: wall boundaries [x_lower, x_upper, y_lower, ..., ...]
            [0.0, 1.0, -0.127, 0.127, 0.0, 0.254]
    :return: 1D array of distances from wall
    """

    above_x = positions['position_x'] - wall_bounds[0]
    below_x = wall_bounds[1] - positions['position_x']
    above_y = positions['position_y'] - wall_bounds[2]
    below_y = wall_bounds[3] - positions['position_y']
    above_z = positions['position_z'] - wall_bounds[4]
    below_z = wall_bounds[5] - positions['position_z']

    dist_all_walls = np.array([above_x, below_x,
                               above_y, below_y,
                               above_z, below_z])

    return np.min(dist_all_walls, axis=0)
