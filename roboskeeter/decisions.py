
class Decisions:
    def __init__(self, decision_policy, stimulus_memory_n_timesteps):
        self.decision_policy = decision_policy

        self.stimulus_memory_n_timesteps = stimulus_memory_n_timesteps
        self.plume_sighted_ago = 10000000  # a long time ago
        self.last_plume_side_exited = None

        self.make_decision = self._set_decision_policy()


    def _set_decision_policy(self):
        if 'cast' in self.decision_policy:
            policy = self._boolean_decisions
        elif 'surge' in self.decision_policy:
            policy = self._boolean_decisions
        elif 'gradient' in self.decision_policy:
            policy = self._gradient_decisions
        elif 'ignore' in self.decision_policy:
            policy = self._ignore_plume
        else:
            raise ValueError('unk decision policy {}'.format(self.decision_policy))
        return policy

    def _boolean_decisions(self, in_plume, crosswind_velocity):
        if in_plume == True:  # use == instead of "is" because we're using type np.bool
            self.plume_sighted_ago = 0
            plume_signal = 'in'
            if 'surge' in self.decision_policy:
                current_decision = 'surge'
            else:
                current_decision = 'search'
        elif in_plume == False:
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
                else:
                    current_decision = 'search'

        return current_decision, plume_signal

    def _gradient_decisions(self, *_):
        """
        Returns
        -------
        current_decision
            we are always following the gradient in this model
        plume_signal
            tell upstream code to look up plume signal
        """
        plume_signal = 'X'  # magic code to look up gradient
        current_decision = 'ga'

        return current_decision, plume_signal

    def _ignore_plume(self):
        return 'ignore', 0
