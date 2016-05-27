__author__ = 'richard'

import logging
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar, basinhopping

from roboskeeter import experiments

logging.basicConfig(filename='basin_hopping.log', level=logging.DEBUG)

# TODO: save guesses and their scores


    # if PLOTTER is True:
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)
    #
    # if np.any(np.isinf(dkl_scores)):
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)



# def error_plotter(ensemble, guess, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed):
#
#
#
#     print "{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score)
#     logging.info("{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score))
#
#     plt.ioff()
#     f, axarr = plt.subplots(2, sharex=False)
#     axarr[0].plot(aabs_bins[:-1], a_counts_n, label='RoboSkeeter')
#     axarr[0].plot(aabs_bins[:-1], a_observed, c='r', label='experiment')
#     axarr[0].set_title('accel score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[1], HIGH_SCORE))
#     axarr[0].legend()
#
#     axarr[1].plot(vabs_bins[:-1], v_counts_n, label='RoboSkeeter')
#     axarr[1].plot(vabs_bins[:-1], v_observed, c='r', label='experiment')
#     axarr[1].set_title('velocity score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[0], HIGH_SCORE))
#     axarr[1].legend()
#     plt.suptitle('Parameter Guess: {}'.format(guess))
#     plt.show()
#     plt.close()


class Baseline_Bounds(object):
    # ["resitution", "randomF", "damping"]
    def __init__(self, xmin=[0., 1e-7, 1e-7], xmax=[1., 1e-4, 1e-4]):
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.xmin))
        tmax = bool(np.all(x <= self.xmax))

        return tmax and tmin


# def main(x_0=None):
#
#     if x_0 is None:
#         INITIAL_GUESS = [0.1]
#     else:
#         INITIAL_GUESS = x_0


    # ################################################ if minimizing a scalar
    # result = minimize_scalar(
    #     fly_wrapper,
    #     bounds=(0., 1.),
    #     method='bounded',
    #     args=(load_mosquito_kde_data_dicts())
    # )

    # return BEST_GUESS, HIGH_SCORE, result


class FitBaselineModel:
    def __init__(self, initial_guess, n_trajectories = 100):
        self.function = basinhopping

        self.stepsize = 1e-6
        self.temperature = 1e-5
        self.niter = 50  # number of basin hopping iterations, default 100
        self.niter_success = 8  # Stop the run if the global minimum candidate remains the same for this number of iterations

        # init buest guess with very large score
        self.best_guess = None
        self.best_score = 1e10

        self.optimizer = 'SLSQP'  # for multiple vars
        self.plotting = False

        self.parameter_names = ["resitution", "randomF", "damping"]  # # [5e-6, 4.12405e-6, 5e-7]  # TODO fix order

        self.score_weights = {'velocity_x': 1,
                                'velocity_y': 1,
                                'velocity_z': 1,
                                'acceleration_x': 1,
                                'acceleration_y': 1,
                                'acceleration_z': 1,
                                'position_x': 1,
                                'position_y': 1,
                                'position_z': 1,
                                'curvature': 3}

        self.bounds = Baseline_Bounds()

        self.initial_guess = initial_guess
        self.n_trajectories = n_trajectories

        self.reference_data = self._load_reference_ensemble()

        logging.info("""\n ############################################################
        ############################################################
        {date}
        algorithm = {algo}
        n_trajectories = {N}
        Params = {pn}
        Initial Guess = {i}
        stepsize = {ss}
        T = {T}
        niter = {ni}
        niter_success = {nis}
        ############################################################""".format(
            date=datetime.now(),
            algo=self.optimizer,
            N=self.n_trajectories,
            pn=self.parameter_names,
            i=self.initial_guess,
            ss = self.stepsize,
            T = self.temperature,
            ni = self.niter,
            nis = self.niter_success))


        self.result = self.run_optimization()

    def run_optimization(self):
        try:
            result = basinhopping(
                self.simulation_wrapper,
                self.initial_guess,
                stepsize=self.stepsize,
                T=self.temperature,
                niter=self.niter,  # number of basin hopping iterations, default 100
                niter_success=self.niter_success,  # Stop the run if the global minimum candidate remains the same for this number of iterations
                minimizer_kwargs={
                    'method': self.optimizer,
                    "tol": 0.02  # tolerance for considering a basin minimized, set to about the difference between re-scoring
                    # same simulation
                },
                disp=True,
                accept_test=self.bounds)

            return result
        except KeyboardInterrupt:
            print "\n Optimization interrupted! Moving along..."
            return ""

    def simulation_wrapper(self, guess):
        """
        :param bias_scale_GUESS:
        :param mass_GUESS:
            mean mosquito mass is 2.88e-6
        :param damping_GUESS:
            estimated damping was 5e-6, # cranked up to get more noise #5e-6,#1e-6,  # 1e-5
        :return:
        """
        print """
        ##############################################################
        Guess: {}
        ##############################################################""".format(guess)


        resitution, randomF, damping  = guess

        experiment_conditions = {'condition': 'Control',
                             'time_max': 6.,
                             'bounded': True,
                             'plume_model': "None"
                             }

        agent_kwargs = {'is_simulation': True,
                        'random_f_strength': resitution,
                        'stim_f_strength': 0.,
                        'damping_coeff': damping,
                        'collision_type': 'part_elastic',
                        'restitution_coeff': resitution,  # Optimizing this
                        'stimulus_memory_n_timesteps': 1,
                        'decision_policy': 'ignore',  # 'surge_only', 'cast_only', 'cast+surge', 'gradient', 'ignore'
                        'initial_position_selection': 'downwind_high',
                        'verbose': False,
                        'optimizing': True
                        }

        experiment = experiments.start_simulation(self.n_trajectories, agent_kwargs, experiment_conditions)

        combined_score, score_components = experiment.calc_score(score_weights=self.score_weights, reference_data=self.reference_data)  # save on computation by passing the ref data

        log_str = "guess = {}. total score = {}. score components = {}. time = {}".format(guess, combined_score, score_components, datetime.now())
        logging.info(log_str)
        print log_str

        if combined_score < self.best_score:  # move to own func
            self.best_score = combined_score
            self.best_guess = guess
            hs_announcement = "accepted {} as new best guess".format(guess)
            print(hs_announcement)
            logging.info(hs_announcement)

        return combined_score

    def _load_reference_ensemble(self):
        reference_experiment = experiments.load_experiment(experiment_conditions = {'condition': 'Control',
                                 'plume_model': "None",
                                 'time_max': "N/A (experiment)",
                                 'bounded': True
                                 })

        return reference_experiment.observations.get_kinematic_dict()



if __name__ == '__main__':
    initial_guess = [0.1, 6.55599224e-06, 3.63674551e-07]  # ["resitution", "randomF", "damping"]

    O = FitBaselineModel(initial_guess)
    result = O.run_optimization()

    msg = """############################################################
    Trial end. FINAL GUESS: {0}
    SCORE: {1}
    RESULT: {2}
    ############################################################""".format(O.best_guess, O.best_score, result)

    logging.info(msg)
    print msg
