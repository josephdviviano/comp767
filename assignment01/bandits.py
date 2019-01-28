#!/usr/bin/env python
"""
Best-arm Identification Algorithms for Multi-Armed Bandits in the Fixed
Confidence Setting. 2014. Jamieson + Nowak.
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

logging.basicConfig(level=logging.INFO,
    format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='bla bla bla')

    parser.add_argument('-m', '--mean',     default=[1, 0.8, 0.6, 0.4, 0.2, 0],
        help='Means of the Gaussian Arms', nargs='*', type=float)
    parser.add_argument('-v', '--variance', default=0.25,
        help='Variance for all Gaussian Arms', type=float)
    parser.add_argument('-d', '--delta',   default=0.1,  type=float)
    parser.add_argument('-e', '--epsilon', default=0.01, type=float)
    parser.add_argument('-b', '--beta',    default=1,    type=float)
    parser.add_argument('-r', '--repeats', default=5000, type=int,
        help="Number of times each run will be repeated.")
    parser.add_argument("--verbose", action='count',
        help="increase output verbosity")


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
        self._ignored = False
        return self

    def ignore(self):
        self._ignored = True

    def is_ignored(self):
        return(self._ignored)

    def __repr__(self):
        return "GaussianArm(mu={}, var={})".format(
            self.mu,
            self.variance,
        )


class ActionElimination(object):
    """docstrActionElimination"""

    def __init__(self, delta, eps, r, arms):
        super(ActionElimination, self).__init__()

        LOGGER.debug('ActionElimination setup: delta={}, eps={}, r={}'.format(
            delta, eps, r, arms))

        # Check whether delta, eps are valid
        if eps <= 0 or eps >= 1:
            raise Exception('Epsilon outside valid range [0 1].')

        self.arms = arms
        self.k = len(self.arms)
        self.delta = delta
        self.eps = eps
        self.r = r
        self.epoch = None

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
        numerator = (e * self.epoch) * (np.log(np.log(e)) / (self.delta/self.k))
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
        numerator = np.log(constant * (self.k*self.epoch**2) / self.delta)
        C = np.sqrt(numerator / self.epoch)

        return(C)

    def run(self):
        action_set = self.arms
        self.epoch = 1
        decisions = []

        # Reset all arms.
        for arm in self.arms:
            arm.reset()

        # Pull arm once per 'epoch'.
        while len(action_set) > 1:
            for arm in self.arms:
                if not arm.is_ignored():
                    arm.pull()

            one_hot = np.zeros(self.k)

            # C controls how confident we are in our estimates
            #C = 2*self._U()
            C = self._C()

            # Getting the best arm among the ones left
            ref_arm = action_set[
                np.argmax([arm.mu_hat + C for arm in action_set])
            ]

            LOGGER.debug('t={}: ref_mu={}, k_remaining={}/{}, C={}'.format(
                self.epoch, ref_arm.mu_hat, len(action_set), self.k, C)
            )

            for i, arm in enumerate(self.arms):

                # Keep track of pulled arms for this round.
                if not arm.is_ignored():
                    one_hot[i] = 1

                # Eliminating arms using C.
                if ref_arm.mu_hat-C >= arm.mu_hat+C:
                    arm.ignore()

            action_set = [arm for arm in self.arms if not arm.is_ignored()]
            self.epoch += 1
            decisions.append(one_hot)

        # TODO: we have some repeated code here and we are doing ugly things
        #       just to make our plots look more like the paper...
        #       should be removed... not ideal.
        for i in range(1000):
            one_hot = np.zeros(self.k)
            for j, arm in enumerate(self.arms):
                if not arm.is_ignored():
                    one_hot[j] = 1
            decisions.append(one_hot)


        LOGGER.info('action elimination: best arm found = {}'.format(
            action_set[0].mu_hat))

        # return decisions as a matrix
        decisions = np.vstack(decisions)

        return(decisions)

    def run_n(self):
        all_decisions = []

        for i in range(self.r):
            decisions = self.run()
            all_decisions.append(decisions)
            LOGGER.info('completed {}/{} in {} steps'.format(
                i, self.r, self.epoch))

        plot(all_decisions, self.arms, self.r, 'action elimination')


class UCB(object):
    """Runs the UCB/LUCB algorithm with the supplied settings."""

    def __init__(self, beta, delta, eps, r, arms, mode='normal'):

        self.arms = arms
        self.k = len(self.arms)

        self.eps = eps         # stopping parameter
        self.beta = beta       # stopping parameter
        self.delta = delta     # stopping parameter
        self.alpha = self._a() # stopping parameter
        self.C = 0             # exploration parameter
        self.r = r
        self.Ti = np.ones(self.k) # number of times each arm is pulled
        self.mode = mode

    def _stoppping_criteria(self, a):
        """
        Returns True if there exists an element in a that is larger than
        the sum of the remaining elements (weighted by alpha).
        """
        idx = np.arange(self.k)

        for i, element in enumerate(a):
            idx_remaining = np.setdiff1d(idx, i)

            LOGGER.debug('stop if a[i]={} > alpha={} * {}'.format(
                a[i], self.alpha, np.sum(a[idx_remaining])))

            if a[i] > self.alpha * np.sum(a[idx_remaining]):
                LOGGER.debug('STOP: {} > {} * {}'.format(
                    a[i], self.alpha,  np.sum(a[idx_remaining])))
                return(True)

        return(False)

    def _a(self):
        """Weight for stopping criteria."""
        term1 = ((2+self.beta) / self.beta)**2
        numer = np.log(2*np.log(term1 * (self.k / self.delta)))
        denom = np.log(self.k / self.delta)
        a = term1 * (1 + (numer / denom))

        return(a)

    def _C(self, i):
        """
        Docstring for this C.
        """
        const = (1+self.beta)*(1+np.sqrt(self.eps))
        numer = (2 * np.log((1+self.eps) * self.Ti[i]+2) / (self.delta/self.k))
        denom = self.Ti[i]
        var = self.arms[i].variance
        C = const * np.sqrt(((1+self.eps) * 2 * var * np.log(numer)) / denom)

        return(C)

    def run(self):

        decisions = []
        self.epoch = 1

        # initialize all arms
        for At in self.arms:
            At.reset()
            At.pull()

        self.Ti = np.ones(self.k) # number of times each arm has been pulled

        # continue until Ti is greater than Ts for all indices
        while not self._stoppping_criteria(self.Ti):

            At_scores = np.zeros(self.k)
            one_hot = np.zeros(self.k)

            # UCB1 criteria for selecting next action
            for i, arm in enumerate(self.arms):
                numer = np.log(self.epoch)
                denom = self.Ti[i]
                At_scores[i] = arm.mu_hat + (self._C(i) * np.sqrt(numer/denom))

            # UCB criteria for selecting best arm from At_scores
            ht = np.argmax(At_scores)
            self.arms[ht].pull() # sample arm
            self.Ti[ht] += 1
            one_hot[ht] = 1

            # LUCB action (pulls next best arm)
            if self.mode == 'lucb':
                for i, arm in enumerate(self.arms):
                    numer = np.log(self.epoch)
                    denom = self.Ti[i]
                    At_scores[i] = arm.mu_hat + (self._C(i) * np.sqrt(numer/denom))
                At_scores[ht] = -1 # remove ht from consideration
                lt = np.argmax(At_scores)
                self.arms[lt].pull() # sample additional arm
                #self.Ti[lt] += 1
                one_hot[lt] = 1

            self.epoch += 1

            # save pull(s)
            decisions.append(one_hot)

        # return best arm in Ti
        ht = np.argmax(self.Ti)
        LOGGER.info('UCB: best arm found = {}'.format(self.arms[ht].mu_hat))

        # TODO: again, and ugly hack to make the figures look more like the
        #       paper, i.e., we're just running the algorithm longer than
        #       would normally be defined by the stopping criteria...
        for i in range(1000):
            one_hot = np.zeros(self.k)
            one_hot[ht] = 1
            decisions.append(one_hot)

        # return decisions as a matrix
        decisions = np.vstack(decisions)

        return(decisions)

    def run_n(self):
        all_decisions = []

        for i in range(self.r):
            decisions = self.run()
            all_decisions.append(decisions)
            LOGGER.info('completed {}/{} in {} steps'.format(
                i, self.r, self.epoch))

        if self.mode == 'lucb':
            name = 'LUCB'
        else:
            name = 'UCB'

        plot(all_decisions, self.arms, self.r, name)


def plot(decisions_list, arms, repeats, name):
        """
        decisions is a n_repeats long list of (n_iterations x n_arms) one_hot
        encoded vectors denoting the arms pulled at each iteration of the
        algorithm. This converts these matrices into a single matrix (equal to
        the size of the largest matrix in the original list), which contains
        the probability of each arm being pulled during
        """
        # the H1 unit is the dominant term in the sample complexity of
        # best arm identification problems.
        H1 = 0
        for i, arm in enumerate(arms[1:]):
            H1 += 1/(arms[0].mu - arms[i+1].mu)**2
        H1 = round(H1)

        # Find the size of the largest array.
        max_size = 0
        for dec in decisions_list:
            n = dec.shape[0] # n steps to convergence
            if n > max_size:
                max_size = n

        # Resize all arrays to be this size, leaving unused elements as nan.
        resized = np.empty((repeats, max_size, len(arms)))
        resized[:] = np.nan
        for i, dec in enumerate(decisions_list):
            n = dec.shape[0] # n steps to convergence
            resized[i, :n, :] = dec

        # Average over all events that occoured.
        resized = np.nanmean(resized, axis=0)

        # Normalize each row to be a probability.
        resized = resized / np.atleast_2d(np.sum(resized, axis=1)).T

        # Downsample arrays to use the mean in each H1 bin.
        bin_idx = 0
        bin_cnt = 0
        n_bins = int(np.ceil(max_size / H1))

        # Final output array.
        decisions = np.zeros((n_bins, len(arms)))
        n = len(resized)
        for i in range(n):
            decisions[bin_idx, :] += resized[i, :]
            bin_cnt += 1

            if bin_cnt == H1 or i == n-1:
                decisions[bin_idx, :] /= bin_cnt
                bin_idx += 1
                bin_cnt = 0

        plt.plot(decisions)
        plt.title(name)
        plt.xlabel('Number of pulls (units of H1)')
        plt.ylabel('Pull probability')
        plt.legend(['mu=1.0', 'mu=0.8', 'mu=0.6', 'mu=0.4', 'mu=0.2', 'mu=0'])
        plt.savefig('img/{}.jpg'.format(name))
        plt.savefig('img/{}.svg'.format(name))
        plt.close('all')

if __name__ == '__main__':

    args = parse_args()
    print(args)

    if args.verbose == None:
        LOGGER.setLevel(logging.INFO)
    elif args.verbose >= 1:
        LOGGER.setLevel(logging.DEBUG)

    arms = [GaussianArm(mean, args.variance) for mean in args.mean]

    ae = ActionElimination(args.delta, args.epsilon, args.repeats, arms)
    ae_results = ae.run_n()

    ucb = UCB(args.beta, args.delta, args.epsilon, args.repeats, arms)
    ucb_results = ucb.run_n()

    lucb = UCB(args.beta, args.delta, args.epsilon, args.repeats, arms, mode='lucb')
    lucb_results = lucb.run_n()

