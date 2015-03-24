"""
Created on Mon Mar 23 15:07:40 2015

@author: rkp

Generate "mosquito" trajectories using harmonic oscillator equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

## define params
dt = 0.01  # (s)
m = 2.5e-6  # mass (kg)
k = 1e-6  # spring constant (kg/s^2)
beta = 1e-6  # damping force (kg/s) NOTE: if beta is too big, things blow up
f0 = 0.  # random driving force magnitude (N)
Tmax = 50  # maximum flight time (s)

rdetect = 0.02  # distance from which mozzie can detect source (m)


def random_force(f0, dim=2):
    """Generate a random-direction force vector at each timestep with
    uniform magnitude f0."""
    if dim == 2:
        # choose direction
        theta = np.random.uniform(high=2*np.pi)
        # return x and y component of force vector
        return f0*np.array([np.cos(theta), np.sin(theta)])
    else:
        raise NotImplementedError('Too many dimensions!')


def traj_gen(r0, v0, rs=None, k=k, beta=beta, f0=f0):
    """Generate a single trajectory.

    Args:
        r0: initial position (list/array)
        v0: initial velocity (list/array)
        rs: source position (list/array) (set to None if no source)

    Returns:
        t: time vector
        r: trajectory positions at all times (array)
        v: trajectory velocities at all times (array)
        a: trajectory accelerations at all times (array)
        source_found: boolean that is True if source is found
        tfound: time till source found (None if not found)
    """

    ## get dimension
    dim = len(r0)

    ## initialize all arrays
    ts_max = int(np.ceil(Tmax / dt))  # maximum time step
    t = np.arange(0, Tmax, dt)  # why does this use Tmax instead of ts_max? -rd
    r = np.zeros((ts_max, dim), dtype=float)
    v = np.zeros((ts_max, dim), dtype=float)
    a = np.zeros((ts_max, dim), dtype=float)

    ## insert initial position and velocity into r,v arrays
    r[0] = r0
    v[0] = v0

    ## loop through timesteps
    for ts in range(ts_max-1):

        # calculate current force
        force = -k*r[ts] - beta*v[ts] + random_force(f0)  #isn't this a new formula? -rd
        # calculate current acceleration
        a[ts] = force/m

        # update velocity in next timestep
        v[ts+1] = v[ts] + a[ts]*dt

        # update position in next timestep
        r[ts+1] = r[ts] + v[ts+1]*dt  #why not use v[ts]? -rd

        # if source, check if source has been found
        if rs is not None:
            if norm(r[ts+1] - rs) < rdetect:
                source_found = True
                tfound = t[ts]  #should this be t[ts+1]? -rd
                # trim excess times in arrays? -rd
                t = t[:ts+1]
                r = r[:ts+1]
                v = v[:ts+1]
                a = a[:ts+1]
                break  # stop flying
    else:  #why is this else at this indentation? -rd
        source_found = False
        tfound = None

    return t, r, v, a, source_found, tfound


if __name__ == '__main__':
    ## generate eight types of trajectories and plot them
    fig, axs = plt.subplots(4, 2, facecolor='w', figsize=(8, 10), sharex=True,
                            sharey=True, tight_layout=True)

    # with no source
    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=0)
    axs[0, 0].plot(r[:, 0], r[:, 1], lw=2)
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title('no damping/no driving')

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=2e-7)
    axs[1, 0].plot(r[:, 0], r[:, 1], lw=2)
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title('damping/no driving')

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=0, f0=3e-6)
    axs[2, 0].plot(r[:, 0], r[:, 1], lw=2)
    axs[2, 0].set_ylabel('y')
    axs[2, 0].set_title('driving/no damping')

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=2e-7, f0=3e-6)
    axs[3, 0].plot(r[:, 0], r[:, 1], lw=2)
    axs[3, 0].set_ylabel('y')
    axs[3, 0].set_title('driving/damping')

    # with source # TODO: add w/o source, w/source labels to top of columns -rd
    rs = [0.13, 0.01]
    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=0, rs=rs)
    axs[0, 1].plot(r[:, 0], r[:, 1], lw=2)
    axs[0, 1].scatter(rs[0], rs[1], s=25, c='r', lw=0)
    axs[0, 1].set_ylabel('y')
    title_append = ''
    if sf:
        title_append = ' (found source!)'
    axs[0, 1].set_title('no damping/no driving' + title_append)

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=2e-7, rs=rs)
    axs[1, 1].plot(r[:, 0], r[:, 1], lw=2)
    axs[1, 1].scatter(rs[0], rs[1], s=25, c='r', lw=0)
    axs[1, 1].set_ylabel('y')
    title_append = ''
    if sf:
        title_append = ' (found source!)'
    axs[1, 1].set_title('damping/no driving' + title_append)

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=0, f0=3e-6, rs=rs)
    axs[2, 1].plot(r[:, 0], r[:, 1], lw=2)
    axs[2, 1].scatter(rs[0], rs[1], s=25, c='r', lw=0)
    axs[2, 1].set_ylabel('y')
    title_append = ''
    if sf:
        title_append = ' (found source!)'
    axs[2, 1].set_title('driving/no damping' + title_append)

    t, r, v, a, sf, tf = traj_gen([1., 0], [0, 0.4], k=k, beta=2e-7, f0=3e-6, rs=rs)
    axs[3, 1].plot(r[:, 0], r[:, 1], lw=2)
    axs[3, 1].scatter(rs[0], rs[1], s=25, c='r', lw=0)
    axs[3, 1].set_ylabel('y')
    title_append = ''
    if sf:
        title_append = ' (found source!)'
    axs[3, 1].set_title('driving/damping' + title_append)

    for ax in axs.flatten():
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    axs[-1, 0].set_xlabel('x')
    axs[-1, 1].set_xlabel('x')

    plt.draw()
