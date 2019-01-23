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

        while len(action_set) > 1:
            for arm in action_set:
                for _ in range(self.r):
                    arm.pull()

            # C controls how confident we are in our estimates
            C = 2*self._U()

            LOGGER.debug('t={}: k_remaining={}/{}, C={}'.format(
                self.epoch, len(action_set), self.n, C)
            )

            # Getting the best arm among the ones left
            ref_arm = action_set[
                np.argmax([arm.mu_hat + C for arm in action_set])
            ]

            for arm in action_set:
                LOGGER.debug('ref_val/this_arm = {}/{}'.format(
                    ref_arm.mu_hat-C, arm.mu_hat+C)
            )

            # Eliminating arms
            action_set = [
                arm for arm in action_set if ref_arm.mu_hat-C < arm.mu_hat+C
            ]


            self.epoch += 1

        return(action_set)

    def _U(self):
        """
        U is used to calculate this epoch's C, which controls exploration of the
        algorithm (larger U values means the algorithm is more likely to keep
        arms during the action elimination step).
        """
        # lemma 1 of Jamieson + Nowak 2014.
        tmp = (1+self.eps) * self.epoch
        constant = 1+np.sqrt(self.eps)

        # delta is normalized by the original number of arms
        numerator = tmp * np.log(np.log(tmp) / (self.delta/self.n))
        denominator = 2*self.epoch

        LOGGER.debug('constant={}, numerator={}, denominator={}'.format(
            constant, numerator, denominator))

        return(constant * np.sqrt(numerator / denominator))


    #def _U2(self, var):
    #    """
    #    Basic confidence interval...
    #    """
    #    conf = 1.05 # alpha=0.05
    #    np.sqrt(var) / np.log(1+ (self.epoch/))
    #    return(1.05 * np.sqrt(variance))/np.log(1+(arm_count[i]/n))



class UCB(object):
    """Runs the UCB/LUCB algorithm with the supplied settings."""

    def __init__(self, delta, epsilon, r, pulls, n, arms, mode='normal'):

        self.delta = delta
        self.epsilon = epsilon
        self.r = r
        self.pulls = pulls
        self.n = n
        self.arms = arms
        self.K = len(self.arms)
        self.Q = np.zeros((self.n, self.K)) # reward estimated
        self.N = np.ones((self.n, self.K))  # n times each arm is pulled (min=1)

    def run(self):

        Qi = np.random.normal(Q_STAR, 1) # first pull of all arms
        Qi_mean = np.mean(Qi)

        R = np.zeros(pulls-1)

        for pull in range(pulls-1):

            # run n experiments in loop (could vectorize?)
            for i in range(n):

                # square root term is an estimate of uncertianty in estimate of At
                ucb_Q = Q[i, :] + (C * np.sqrt(np.log(pull) / N[i, :]))
                At = np.argmax(ucb_Q)

                # reward for this pull dependent on action At
                R[pull] = np.random.normal(Q_STAR[At], 1)

                N[i, At] += 1
                Q[i, At] = Q[i, At] + (R[pull]-Q[i, At]) / N[i, At]

        R_mean = np.mean(R, axis=1)


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

    #ucb = UCB(args.delta, args.epsilon, args.r, args.p, args.n, arms)
    #lucb = UCB(args.delta, args.epsilon, args.r, args.p, args.n, arms, mode='lucb')


