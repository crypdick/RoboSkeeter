
class Decisions:
    def __init__(self, plume, decision_policy, stimulus_memory_n_timesteps):
        self.plume = plume
        self.decision_policy = decision_policy

        self.stimulus_memory_n_timesteps = stimulus_memory_n_timesteps
        self.plume_sighted_ago = 10000000  # a long time ago
        self.last_plume_side_exited = None

        if 'cast' in decision_policy or 'surge' in decision_policy:
            # sanity check
            if plume.plume_model != 'Boolean':
                raise TypeError('expecting a boolean plume')
            self.make_decision = self._boolean_decisions
        elif 'gradient' in decision_policy:
            # sanity check
            if plume.plume_model != 'TimeaAvg':  # TODO add other types

            self.make_decision = self._gradient_decisions


    def _boolean_decisions(self, in_plume, crosswind_velocity):
        current_decision = ''  # reset

        if in_plume is True:
            self.plume_sighted_ago = 0
            plume_interaction = 'in'
            if 'surge' in self.decision_policy:
                current_decision = 'surge'
        elif in_plume is False:
            self.plume_sighted_ago += 1
            if self.plume_sighted_ago == 1:  # we just exited the plume
                # if our y velocity is negative, we just exited to the left. otherwise, to the right.
                if crosswind_velocity < 0:
                    plume_interaction = 'exit left'
                    self.last_plume_side_exited = 'l'
                else:
                    plume_interaction = 'exit right'
                    self.last_plume_side_exited = 'r'
            else:
                plume_interaction = 'out'
                if 'cast' in self.decision_policy:
                    if self.plume_sighted_ago <= self.stimulus_memory_n_timesteps:  # we were in the plume recently
                        if self.last_plume_side_exited == 'l':
                            current_decision = 'cast_r'
                        else:  # 'r'
                            current_decision = 'cast_l'
                else:
                    current_decision = 'search'

        return current_decision, plume_interaction


    def _gradient_decisions(self, position):
            plume_interaction = self.plume.get_nearest_gradient(position)
            # TODO
            # if gradient is above thresh, 'ascend'
            #else: search
            # surge up gradient
        else:
            raise TypeError

        return current_decision, plume_interaction
