#!/usr/bin/env python
"""
Best-arm Identification Algorithms for Multi-Armed Bandits in the Fixed
Confidence Setting. 2014. Jamieson + Nowak.
"""
import argparse
import logging
import numpy as np
import os
import sys
import warnings


logging.basicConfig(level=logging.INFO,
    format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='bla bla bla')

    parser.add_argument(
        '-m', '--mean', default=[1, 0.8, 0.6, 0.4, 0.2, 0], nargs='*', type=float,
        help='Means of the Gaussian Arms'
    )

    parser.add_argument(
        '-v', '--variance', default=0.25, type=float,
        help='Variance for all Gaussian Arms'
    )

    parser.add_argument(
        '-d', '--delta', default=0.1, type=float
    )

    parser.add_argument(
        '-e', '--epsilon', default=0.5, type=float
    )

    parser.add_argument(
        '-r', '--repeats', default=1, type=int,
        help="Number of times the arms will be sampled at each epoch"
    )

    parser.add_argument(
        '-p', '--pulls', default=100, type=int,
        help='Maximum number of pulls'
    )

    parser.add_argument(
        '--n', '--nexp', default=5000, type=int,
        help='number of experiments to run'
    )

    parser.add_argument(
        "--verbose", action='count', help="increase output verbosity"
    )


    return(parser.parse_args())


class GaussianArm(object):
    """docstring for GaussianArm"""

    def __init__(self, mu, variance):
        super(GaussianArm, self).__init__()
        self.mu = mu
        self.variance = variance
        self.std = variance ** 0.5
        self.reset()

    @property
    def count(self):
        return self._count

    @property
    def mu_hat(self):
        # Using cumulative average instead of storing all pulled values
        return self._sum_pull / self.count

    @property
    def variance_hat(self):
        # Using cumulative average instead of storing all pulled values
        v = self._sum_pull_square / self.count
        v -= (self._sum_pull / self.count) ** 2
        return v

    def pull(self):
        value = self.std * np.random.randn() + self.mu
        self._sum_pull += value
        self._sum_pull_square += value ** 2
        self._count += 1
        return value

    def reset(self):
        self._count = 0
        self._sum_pull = 0
        self._sum_pull_square = 0
        return self

    def __repr__(self):
        return "GaussianArm(mu={}, var={})".format(
            self.mu,
            self.variance,
        )


class ActionElimination(object):
    """docstring for ActionElimination"""

    def __init__(self, delta, eps, r, arms):
        super(ActionElimination, self).__init__()

        LOGGER.debug('ActionElimination setup: delta={}, eps={}, r={}'.format(
            delta, eps, r, arms))

        # Check whether delta, eps are valid
        if eps <= 0 or eps >= 1:
            raise Exception('Epsilon outside valid range [0 1].')

        delta_max = np.log(1+eps) / np.e

        if delta <= 0 or delta >= delta_max:
            raise Exception('Delta outside valid range [0 {}].'.format(
                delta_max)
            )

        self.delta = delta
        self.eps = eps
        self.r = r
        self.arms = arms
        self.n = len(self.arms)
        self.epoch = None

    def run(self):
        action_set = self.arms
        self.epoch = 1

        # pull arm once per 'epoch'
        while len(action_set) > 1:
            for arm in action_set:
                arm.pull()

            # C controls how confident we are in our estimates
            #C = 2*self._U()
            C = self._C()


            # Getting the best arm among the ones left
            ref_arm = action_set[
                np.argmax([arm.mu_hat + C for arm in action_set])
            ]

            LOGGER.debug('t={}: ref_mu={}, k_remaining={}/{}, C={}'.format(
                self.epoch, ref_arm.mu_hat, len(action_set), self.n, C)
            )

            # Eliminating arms using C
            action_set = [
                arm for arm in action_set if ref_arm.mu_hat-C < arm.mu_hat+C
            ]

            self.epoch += 1

        LOGGER.info('action elimination: best arm found = {}'.format(
            action_set[0].mu_hat))

        return(action_set)

    def _U(self):
        """
        U is used to calculate this epoch's C, which controls exploration of the
        algorithm (larger U values means the algorithm is more likely to keep
        arms during the action elimination step).

        TODO: depricate. converges very slowly, or not at all.
        """
        warnings.warn(
            "_U is deprecated, use _C instead",
            DeprecationWarning
        )

        # lemma 1 of Jamieson + Nowak 2014, with typo fixed!
        constant = 1+np.sqrt(self.eps)

        # delta is normalized by the original number of arms
        e = 1+self.eps
        numerator = (e * self.epoch) * (np.log(np.log(e)) / (self.delta/self.n))
        denominator = 2*self.epoch

        LOGGER.debug('constant={}, numerator={}, denominator={}'.format(
            constant, numerator, denominator))

        return(constant * np.sqrt(numerator / denominator))

    def _C(self):
        """
        Successive elimination method from section 4. C controls the
        exploration of the algorithm (larger C values means the algorithm is
        more likely to keep arms during the action elimination step).
        """
        constant = (np.pi**2)/3
        numerator = np.log(constant * (self.n*self.epoch**2) / self.delta)
        C = np.sqrt(numerator / self.epoch)

        return(C)


class UCB(object):
    """Runs the UCB/LUCB algorithm with the supplied settings."""

    def __init__(self, delta, eps, r, arms, mode='normal'):

        self.delta = delta
        self.eps = eps
        self.beta = beta
        self.lamb = lamb
        self.r = r
        self.arms = arms

        self.k = len(self.arms)
        self.Q = np.zeros((self.r, self.K)) # reward estimated
        self.N = np.ones((self.r, self.K))  # n times each arm is pulled (min=1)

    def _compare_vecs(self, a, b):
        n_less =


    def run(self):

        action_set = self.arms
        self.epoch = 1

        # initialize all arms
        for arm in action_set:
            arm.pull()

        Ti = np.zeros(self.k) # number of times each arm has been pulled
        Ts = np.ones(self.k)  # stopping condition

        #sigma = 0.25**2 # TODO: scale parameter, where does it come from?

        # for calculating It at each iteration
        #const = (1+self.beta) * (1+np.sqrt(self.eps))
        #num_const = 2*sigma * (1+self.eps)

        # continue until Ti is greater than Ts for all indices
        while np.sum(Ti < Ts) > 0:

            It_candidates = np.zeros(self.k)

            for i, arm in enumerate(arms):
                arm.pull() # sample arm

                # calculated in loop as it depends on Ti[i]
                #num = num_const * np.log(np.log((1+self.eps)*Ti[i])/self.delta)

                # lilUCB formula from Jamieson et al 2014
                #It_candidates[i] = (arm.mu_hat + const) * np.sqrt(num / Ti[i])

            It = action_set[np.argmax(It_candidates)]
            Ti[np.argmax(It_candidates)] += 1


if __name__ == '__main__':

    args = parse_args()

    print(args)

    # logging
    if args.verbose == None:
        LOGGER.setLevel(logging.INFO)
    elif args.verbose >= 1:
        LOGGER.setLevel(logging.DEBUG)

    arms = [GaussianArm(mean, args.variance) for mean in args.mean]

    ae = ActionElimination(args.delta, args.epsilon, args.repeats, arms)
    ae_results = ae.run()

    ucb = UCB(args.delta, args.epsilon, args.repeats, arms)
    ucb_results = ucb.run()

    #lucb = UCB(args.delta, args.epsilon, args.r, args.p, args.n, arms, mode='lucb')


