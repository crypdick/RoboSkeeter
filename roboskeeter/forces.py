__author__ = 'richard'
import numpy as np
from roboskeeter.math import math_toolbox


class Forces():
    def __init__(self, randomF_strength, stimF_strength, stimulus_memory, decision_policy):
        # TODO export stim stuff
        self.randomF_strength = randomF_strength
        self.stimF_strength = stimF_strength
        self.stimulus_memory = stimulus_memory
        self.decision_policy = decision_policy

    def randomF(self):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.
        """
        # TODO: make randomF draw from the canonical eqn for random draws Rich taught you
        ends = math_toolbox.gen_symm_vecs(3)
        force = self.randomF_strength * ends

        return force
