__author__ = 'richard'

import numpy as np

from scripts import math_sorcery


class Forces():
    def __init__(self, randomF_strength, stimF_strength, stimulus_memory, decision_policy):
        # TODO: save coefficients here
        self.randomF_strength = randomF_strength
        self.stimF_strength = stimF_strength
        self.stimulus_memory = stimulus_memory
        self.decision_policy = decision_policy

    def randomF(self, dim=3, kind='constant'):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.

        Args:
            rf: random force distribution exponent (float)

        Returns:
            random force x and y components (array)
        """
        if kind is 'lognorm':
            MAG_THRESH = 3e-4

            ends = math_sorcery.gen_symm_vecs(3)
            # following params were fitted from the Dickinson fligt data
            mu = 0.600023812816
            sigma = 0.719736466122
            # scale = 1.82216219069
            mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
            if mag * self.randomF_strength > MAG_THRESH:  # filter out huge magnitudes
                force = ends * MAG_THRESH
            else:
                force = mag * self.randomF_strength * ends

        elif kind is 'constant':
            ends = math_sorcery.gen_symm_vecs(3)
            force = self.randomF_strength * ends

        return force

    def stimF(self, args):
        """given force direction and strength, return a force vector
        Args:
        experience, tsi, tsi_last_stighted
        TODO: if less than last_sighted, keep on  
        """
        if self.decision_policy is 'cast_only':
            force = self._stimF_cast_only(*args)
        elif self.decision_policy is 'surge_only':
            force = self._stimF_surge_only(*args)
        elif self.decision_policy is 'cast+surge':
            raise NotImplementedError
        else:
            print "stimF error", kind
            force = np.array([0., 0., 0.])

        return force

    def _stimF_cast_only(self, tsi, tsi_plume_last_sighted, plume_interaction_history):
        memory = plume_interaction_history[tsi - self.stimulus_memory:tsi]
        if tsi - tsi_last_sighted >= self.stimulus_memory:  # no stimulus recently
            force = np.array([0., 0., 0.])
        else:
            memory = np.fliplr([memory])[0]
            for experience in nd.iter(memory):
                if experience is 'outside':
                    pass
                if experience is 'inside':
                    pass
        if experience is 'outside':
            return np.array
        elif experience in 'staying':
            return np.array([self.stimF_strength, 0., 0.])  # surge while in plume
        elif "Exit left" in experience:
            return np.array([0., self.stimF_strength, 0.])
        elif "Exit right" in experience:
            return np.array([0., -self.stimF_strength, 0.])

        return force

    def _stimF_surge_only(self, tsi, tsi_plume_last_sighted, plume_interaction_history):
        if plume_interaction_history[tsi] is 'inside':
            force = np.array([self.stimF_strength, 0., 0.])
        else:
            force = np.array([0., 0., 0.])

        return force
