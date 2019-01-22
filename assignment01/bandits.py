#!/usr/bin/env python

import argparse
import numpy as np
import sys

# true expected values of each arm
Q_STAR = [1, 0.8, 0.6, 0.4, 0.2, 0]
K = len(Q_STAR)

# constant exploration value for now
C = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description='bla bla bla')

    parser.add_argument(
        '-m',
        '--mean',
        help='Means of the Gaussian Arms',
        default=[1, 0.8, 0.6, 0.4, 0.2, 0],
        nargs='*',
        type=float
    )

    parser.add_argument(
        '-v',
        '--variance',
        help='Variance for all Gaussian Arms',
        default=0.25,
        type=float
    )

    parser.add_argument(
        '-d',
        '--delta',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '-e',
        '--epsilon',
        default=0.01,
        type=float
    )

    parser.add_argument(
        '-r',
        default=1,
        type=int,
        help="Number of times the arms will be sampled at each epoch"
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
        # Using commulative average instead of storing all pulled values
        return self._sum_pull / self.count

    @property
    def variance_hat(self):
        # Using commulative average instead of storing all pulled values
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

    def __init__(self, delta, epsilon, r, arms):
        super(ActionElimination, self).__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.r = r
        self.arms = arms

    def run(self):
        action_set = self.arms
        epoch = 1
        while len(action_set) > 1:
            for arm in action_set:
                for _ in range(self.r):
                    arm.pull()

            # As it stands in the paper, this seems to be the same thing
            # for every arms. If it's a constant for every arm, then it is
            # essentially useless.
            C = 2 * U(epoch, self.delta / len(self.arms), self.epsilon)

            # Getting the best arm among the ones left
            reference_arm = action_set[
                np.argmax([arm.mu_hat + C for arm in action_set])
            ]

            # Eliminating arms
            action_set = [arm for arm in action_set if reference_arm.mu_hat - C < arm.mu_hat + C]
            epoch += 1
        return action_set


def U(t, delta, epsilon):
    tmp1 = 1 + np.sqrt(epsilon)
    tmp2 = (1 + epsilon) * t

    val = tmp1 * np.sqrt(
        (
            tmp2 * np.log(
                np.log(tmp2) / delta
            )
        ) / (2 * t)
    )
    return val



def ae(pulls, n):
    """
    Runs the Action Elimination algorithm with the supplied settings.
    """

    print('ae')


def ucb(pulls, n, mode='normal'):
    """
    Runs the ucb/lucb algorithm with the supplied settings.
    """

	Q = np.zeros((n, K)) # reward estimated
	N = np.ones((n, K))  # number of times each arm was pulled (min = 1)

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


def main(args):
    """
    Runs the appropriate k-armed bandit algorithm defined for the specified
    number of pulls. These results are then passed to a plotting function.
    """

    if args.method == 'ae':
        ae(args.pulls, args.n)
    elif args.method == 'ucb':
        ucb(args.pulls, args.n)
    elif args.method == 'lucb':
        ucb(args.pulls, args.n, mode='lucb')
    else:
        raise Exception('method supplied is not valid: {ae, ucb, lucb}')


if __name__ == '__main__':

    args = parse_args()
    print(args)
    arms = [GaussianArm(mean, args.variance) for mean in args.mean]
    ae = ActionElimination(args.delta, args.epsilon, args.r, arms)
    print(ae.run())

    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, help="{ae, ucb, lucb}")
    parser.add_argument('--pulls', type=int, default=100, help='maximum number of pulls')
    parser.add_argument('--n', type=int, default=5000, help='number of experiments to run')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(e)
        sys.exit(1)

