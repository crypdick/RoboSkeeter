class Behavior(object):
    # TODO: implement behavior class
    # TODO: implement plume boundary interaction
    # TODO: implement plume memory
    def __init__(self, decision_policy):
        self.decision_policy = decision_policy



class Gradient_Following(Behavior):
    pass # TODO implement gradient following

def _plume_interaction(self, tsi, in_plume, velocity_y_now, last_triggered):
    """
    out2out - searching
     (or orienting, but then this func shouldn't be called)
    out2in - entering plume
    in2in - staying
    in2out - exiting
        {Left_plume Exit left, Left_plume Exit right
        Right_plume Exit left, Right_plume Exit right}
    """
    # TODO: only run this stuff if running a relevant decision policy
    current_in_plume, past_in_plume = in_plume[tsi], in_plume[tsi - 1]

    if tsi == 0:  # always start searching
        state = 'outside'
    elif current_in_plume == False and past_in_plume == False:
        # we are not in plume and weren't in last ts
        state = 'outside'
    elif current_in_plume == True and past_in_plume == False:
        # entering plume
        state = 'inside'
    elif current_in_plume == 1 and past_in_plume == True:
        # we stayed in the plume
        state = 'inside'
    elif current_in_plume == False and past_in_plume == True:
        # exiting the plume
        if velocity_y_now <= 0:
            state = 'Exit left'
        else:
            state = 'Exit right'
    else:
        raise Exception("This error shouldn't ever run")

    if state is 'outside':
        pass
    elif state is 'inside':
        last_triggered['stimulus'] = tsi
    elif 'Exit' in state:
        last_triggered['exit'] = tsi

    return state, last_triggered


    def stimF(self, kwargs):
        """given force direction and strength, return a force vector
        decision policies: 'cast_only', 'surge_only', 'cast+surge'

        TODO: review this func to make sure castsurge would work
        """
        force = np.array([0., 0., 0.])

        if 'cast' in self.decision_policy:
            force += self._cast(kwargs)
        if 'surge' in self.decision_policy:
            force += self._surge_upwind(kwargs)
        if 'gradient' in self.decision_policy:
            force += self._surge_up_gradient(kwargs)
        if 'ignore' in self.decision_policy:
            pass

        return force

    def _cast(self, kwargs):
        # TODO: review cast
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']
        triggered_tsi = kwargs['triggered_tsi']

        empty = np.array([0., 0., 0.])  # FIXME naming
        # TODO: check if this is updating triggers, and if it should be
        inside_ago = abs(tsi - triggered_tsi['stimulus'])
        exit_ago = abs(tsi - triggered_tsi['exit'])
        cast_strength = self.stimF_strength / 10
        if tsi == 0:
            return empty  # FIXME naming
        elif inside_ago < exit_ago:  # if we re-encounter the plume, stop casting
            return empty
        elif exit_ago <= self.stimulus_memory:  # stimulus encountered recently
            # print "we have a memory!"
            # print "currently {tsi}, last {last}, difference {diff}".format(tsi=tsi, last=last_triggered['exit'], diff=exit_ago)
            experience = plume_interaction_history[tsi - exit_ago]
            # if experience in 'outside':
            #     pass # keep going back
            # elif experience is 'inside':
            #     pass # keep going back
            if experience == 'Exit left':
                return np.array([0., cast_strength, 0.])  # cast right
            elif experience == 'Exit right':
                return np.array([0., -cast_strength, 0.])  # cast left
            else:
                # print "valueerror! experience", experience, "tsi", tsi
                # print experience == 'Exit right', experience
                raise ValueError('no such experience known: {}'.format(experience))
                # except ValueError:
                #     print "tsi", tsi, "memory", memory[:tsi], plume_interaction_history
                # except TypeError:
                #     print "memory type", memory, type(memory)


        else:  # no recent memory of stimulus
            current_experience = plume_interaction_history[tsi]
            if current_experience in ['outside', 'inside']:
                force = empty
            else:
                print "plume_interaction_history", plume_interaction_history, plume_interaction_history[:tsi]
                print "current_experience", current_experience
                raise ValueError("no such experience {} at tsi {}".format(current_experience, tsi))

            return force

    def _surge_upwind(self, kwargs):
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']

        if plume_interaction_history[tsi] is 'inside':
            force = np.array([self.stimF_strength, 0., 0.])
        else:
            force = np.array([0., 0., 0.])

        return force

def _surge_up_gradient(self, kwargs):
    """gradient vector norm * stimF strength"""
    df = kwargs['gradient']

    scalar = self.stimF_strength
    vector = df[['gradient_x', "gradient_y", "gradient_z"]].values
    force = scalar * vector

    # stimF here is proportional to norm of gradient. in order to avoid huge stimFs, we put a ceiling on the
    # size of stimF
    ceiling = 1e-5  # TODO: parameterize in function call
    norm = np.linalg.norm(force)
    if norm > ceiling:
        force *= 1e-5 / norm

    # catch problems in stimF
    if np.isnan(force).any():
        raise ValueError("Nans in stimF!! {} {}".format(force, vector))
    if np.isinf(force).any():
        raise ValueError("infs in stimF! {} {}".format(force, vector))

    return force
#