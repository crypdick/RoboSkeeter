
class Decisions:
    def __init__(self, plume, decision_policy, stimulus_memory_n_timesteps):
        # we take plume as input for the gradient ascent. however, this is inelegant. FIXME
        self.plume = plume
        self.decision_policy = decision_policy

        self.stimulus_memory_n_timesteps = stimulus_memory_n_timesteps
        self.plume_sighted_ago = 10000000  # a long time ago
        self.last_plume_side_exited = None

        if 'cast' in decision_policy or 'surge' in decision_policy:
            # sanity check
            if plume.plume_model != 'Boolean':
                raise TypeError('expecting a boolean plume model')
            self.make_decision = self._boolean_decisions
        elif 'gradient' in decision_policy:
            # sanity check
            if plume.plume_model != 'TimeaAvg':  # TODO add other types
                raise TypeError("expecting gradient plume model")
            self.make_decision = self._gradient_decisions


    def _boolean_decisions(self, in_plume, crosswind_velocity):
        if in_plume is True:
            self.plume_sighted_ago = 0
            plume_signal = 'in'
            if 'surge' in self.decision_policy:
                current_decision = 'surge'
            else:
                current_decision = 'search'
        elif in_plume is False:
            self.plume_sighted_ago += 1
            if self.plume_sighted_ago == 1:  # we just exited the plume
                # if our y velocity is negative, we just exited to the left. otherwise, to the right.
                if crosswind_velocity < 0:
                    plume_signal = 'exit_l'
                    self.last_plume_side_exited = 'l'
                    current_decision = 'cast_r'
                else:
                    plume_signal = 'exit_r'
                    self.last_plume_side_exited = 'r'
                    current_decision = 'cast_l'
            else:  # been outside plume at least a couple timesteps
                plume_signal = 'out'
                if 'cast' in self.decision_policy:
                    if self.plume_sighted_ago <= self.stimulus_memory_n_timesteps:  # we were in the plume recently
                        if self.last_plume_side_exited == 'l':
                            current_decision = 'cast_r'
                        else:  # 'r'
                            current_decision = 'cast_l'
                else:  # haven't seen the plume in a while
                    current_decision = 'search'

        return current_decision, plume_signal


    def _gradient_decisions(self, position):
        """

        Parameters
        ----------
        position
            current position. used to lookup gradient

        Returns
        -------
        current_decision
            we are always following the gradient in this model
        plume_signal
            the gradient
        """
        plume_signal = self.plume.get_nearest_gradient(position)
        current_decision = 'ga'

        return current_decision, plume_signal
