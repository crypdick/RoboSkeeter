__author__ = 'richard'
import numpy as np
from roboskeeter.math import math_toolbox


class Flight():
    def __init__(self, random_f_strength, stim_f_strength, damping_coeff):
        self.random_f_strength = random_f_strength
        self.stim_f_strength = stim_f_strength  # TODO: separate surge strength, cast strength, gradient strenght
        self.damping_coeff = damping_coeff
        self.max_stim_f = 1e-5  # putting a maximum value on the stim_f

    def random(self):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.
        """
        # TODO: make randomF draw from the canonical eqn for random draws Rich taught you
        ends = math_toolbox.gen_symm_vecs(3)
        force = self.random_f_strength * ends

        return force

    def stimulus(self, decision, plume_signal):
        force = np.array([0., 0., 0.])

        if decision is 'search':
            pass  # there is no stimulus_f in the absence of stimulus
        elif decision is 'ga':
            force += self.surge_up_gradient(plume_signal)
        elif decision is 'surge':
            force += self.surge_upwind()
        elif 'cast' in decision:
            force += self.cast(decision)
        elif decision == 'ignore':
            pass
        else:
            raise LookupError('unknown decision {}'.format(decision))

        return force

    def calc_forces(self, current_velocity, decision, plume_signal):
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        random_f = self.random()

        stim_f = self.stimulus(decision, plume_signal)

        ################################################
        # calculate total force
        ################################################
        total_f = -self.damping_coeff * current_velocity + random_f + stim_f
        ###############################

        return stim_f, random_f, total_f

    def cast(self, decision):
        """

        Parameters
        ----------
        decision
            cast_l or cast_r

        Returns
        -------
        force
            the appropriate cast force
        """
        cast_f = self.stim_f_strength
        if 'l' in decision:  # need to cast left
            cast_f *= -1.
        else:
            pass
        force = np.array([0., cast_f, 0.])

        return force

    def surge_upwind(self):
        force = np.array([self.stim_f_strength, 0., 0.])

        force = self._shrink_huge_stim_f(force)

        return force

    def surge_up_gradient(self, gradient):
        """

        Parameters
        ----------
        gradient
            the current plume gradient

        Returns
        -------
        force
            the stimulus force to ascend the gradient, properly scaled
        """

        scalar = self.stim_f_strength
        force = scalar * gradient

        force = self._shrink_huge_stim_f(force)

        # catch bugs in gradient multiplication
        if np.isnan(force).any():
            raise ValueError("Nans in gradient force!! force = {} gradient = {}".format(force, gradient))
        if np.isinf(force).any():
            raise ValueError("infs in gradient force! force = {} gradient = {}".format(force, gradient))

        return force

    def _shrink_huge_stim_f(self, force):
        norm = np.linalg.norm(force)
        if norm > self.max_stim_f:
            force *= self.max_stim_f / norm  # shrink force to maximum allowed value

        return force

