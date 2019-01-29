

"""
Implement and compare empirically the performance of value iteration,
policy iteration and modified policy iteration. Modified policy iteration is a
simple variant of policy iteration in which the evaluation step is only partial
(that is, you will make a finite number of backups to the value function before
an improvement step). You can consult the Puterman (1994) textbook for more
information. You should implement your code in matrix form. Provide your code
as well as a summary of the results, which shows, as a function of the number
of updates performed, the true value of the greedy policy computed by your
algorithm, from the bottom left state and the bottom right state. That is,
you should take the greedy policy currently considered by your algorithm and
compute its exact value.

To test your algorithm, use a grid world in which, at each time step, your
actions move in the desired direction w.p. p and in a random direction w.p.
(1-p). The grid is empty, of size n × n. There is a positive reward of +10 in
the upper right corner and a positive reward of +1 in the upper left corner.
All other rewards are 0. You need to test your algorithm with two different
values of p (0.9 and 0.7) and with two different sizes of gird
(n = 5 and n = 50). Explain what you see in these results.
"""

from collections import namedtuple
import numpy as np
from enum import Enum
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')


State = namedtuple('State', ['i', 'j', 'reward', 'terminal'])


def parse_args():
    parser = argparse.ArgumentParser(description='bla bla bla')

    parser.add_argument(
        '--size',
        help="Size of the grid",
        default=5,
        type=int
    )

    parser.add_argument(
        '-p', '--prob',
        help="Probability of a successful action",
        default=0.7,
        type=float
    )

    parser.add_argument(
        '--iteration',
        help="Probability of a successful action",
        required=True,
        choices=['policy', 'value', 'modified']
    )

    parser.add_argument(
        '-t', '--theta',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '-d', '--discount',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '-k',
        type=int
    )

    return parser.parse_args()


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def print_grid(values):
    n = len(values)
    m = len(values[0])

    out = ""
    out += "┌" + "───┬" * (m - 1) + "───┐\n"
    for i in range(n):
        out += "│"
        for j in range(m):
            out += "{:^3s}│".format(str(values[i][j]))
        if i != (n - 1):
            out += "\n├" + "───┼" * (n - 1) + "───┤\n"
    out += "\n└" + "───┴" * (m - 1) + "───┘"
    return out


def print_policy(policy):
    "⇠⇢⇡⇣"
    up = " ⇡ "
    down = " ⇣ "
    left = "⇠  "
    right = "  ⇢"
    left_right = "⇠ ⇢"

    n = policy.grid.n
    out = "┌" + "───┬" * (n - 1) + "───┐\n"
    for i in range(n):
        out += "│"
        for j in range(n):
            state = policy.grid[i, j]
            if policy[state][Action.UP] > 0:
                out += up + "│"
            else:
                out += "   │"
        out += "\n│"
        for j in range(n):
            state = policy.grid[i, j]
            if policy[state][Action.LEFT] > 0 and policy[state][Action.RIGHT] > 0:
                out += left_right + "│"
            elif policy[state][Action.LEFT] > 0:
                out += left + "│"
            elif policy[state][Action.RIGHT] > 0:
                out += right + "│"
            else:
                out += "   │"
        out += "\n│"
        for j in range(n):
            state = policy.grid[i, j]
            if policy[state][Action.DOWN] > 0:
                out += down + "│"
            else:
                out += "   │"
        out += "\n"
        if i != (n - 1):
            out += "├" + "───┼" * (n - 1) + "───┤\n"

    out += "└" + "───┴" * (n - 1) + "───┘"
    return out


class Grid(object):
    """docstring for Grid"""

    def __init__(self, n):
        super(Grid, self).__init__()
        self.n = n
        self._grid = np.empty((n, n), dtype=np.object)

        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    state = State(i, j, 1, True)
                elif i == 0 and j == (n - 1):
                    state = State(i, j, 10, True)
                else:
                    state = State(i, j, 0, False)
                self._grid[i, j] = state

    @property
    def rewards(self):
        values = np.empty_like(self._grid, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                values[i, j] = self[i, j].reward
        return values

    def neighbours(self, state):
        return {
            Action.UP: self[max(0, state.i - 1), state.j],
            Action.RIGHT: self[state.i, min(self.n - 1, state.j + 1)],
            Action.DOWN: self[min(self.n - 1, state.i + 1), state.j],
            Action.LEFT: self[state.i, max(0, state.j - 1)]
        }

    def __getitem__(self, X):
        return self._grid[X]

    def __iter__(self):
        for i in range(self.n):
            for j in range(self.n):
                yield(self[i, j])

    def __repr__(self):
        return print_grid(self.rewards)


class Policy(object):
    """docstring for Policy"""

    def __init__(self, grid):
        super(Policy, self).__init__()
        self.grid = grid
        self._policy = {}
        for s in grid:
            prob = 0.25
            if s.terminal:
                prob = 0
            self._policy[s] = {
                Action.UP: prob,
                Action.RIGHT: prob,
                Action.DOWN: prob,
                Action.LEFT: prob
            }

    def __getitem__(self, state):
        return self._policy[state]

    def __len__(self):
        return len(self._policy)

    def as_matrix(self):
        #  If we want to be more dynamic we should check how many actions
        # are actually allowed across all states
        pi = np.zeros((len(self), 4))
        #  Iterating of the grid to make sure we always iteration the same way
        for i, state in enumerate(self.grid):
            for action, prob in self[state].items():
                pi[i, action.value] = prob
        return pi

    def update(self, pi):
        for i, state in enumerate(self.grid):
            for j, prob in enumerate(pi[i]):
                if state.terminal:
                    prob = 0
                self._policy[state][Action(j)] = prob


class Environment(object):
    """docstring for Environment"""

    def __init__(self, policy, p):
        super(Environment, self).__init__()
        self.policy = policy
        self._grid = policy.grid
        self._state_idx = {state: i for i, state in enumerate(policy.grid)}
        self.p = p

    @property
    def P_a(self):
        k = len(self.policy)

        # If we want to be a bit more dynamic we should check how many
        # action the policy allows.
        t = np.zeros((k, k, 4))
        # For every possible state s
        for state, i in self._state_idx.items():
            # For every allowed action starting from state s
            neighbours = self._grid.neighbours(state)
            allowed_action = self.policy[state]
            num_actions = len(allowed_action)
            for action, action_prob in allowed_action.items():
                k = action.value
                for _action in allowed_action:
                    state_prime = neighbours[_action]
                    j = self._state_idx[state_prime]

                    # taking the action may take us to state prime with
                    # probability p
                    if action == _action:
                        t[i, j, k] += self.p * action_prob
                    # or get to a random direction with probability (1 - p)
                    else:
                        # The random direction is encoded by dividing (1 - p)
                        # by the number of other directions
                        p = (1 - self.p) / (num_actions - 1) * action_prob
                        t[i, j, k] += p
        return t

    @property
    def P(self):
        return self.P_a.sum(axis=2)

    @property
    def rewards(self):
        return np.array([s.reward for s in self._state_idx])[:, None]


class PolicyIteration(object):
    """docstring for PolicyIteration"""

    def __init__(self, env, policy, discount, theta=0.0001, k=float("inf")):
        super(PolicyIteration, self).__init__()
        self.env = env
        self.policy = policy
        self.theta = theta
        self.discount = discount

        #  This parameter is to use Modified Policy Iteration
        self.k = k

    def evaluation(self):
        # Make sure it starts higher than the stopping condition
        delta = self.theta * 10.
        V = np.zeros((len(policy), 1))
        # pbar = tqdm(desc="delta {:.5f}".format(delta), leave=False)
        epoch = 0
        n = self.policy.grid.n
        values = []
        while delta > self.theta and epoch < self.k:
            # Saving values of bottom left and right corner
            # for plotting purposes
            _V = V.reshape((n, n))
            values.append((_V[-1, 0], _V[-1, -1]))
            v = V.copy()
            V = np.dot(env.P, self.env.rewards + V * self.discount)
            delta = np.abs(v - V).max()
            # pbar.update(1)
            # pbar.set_description("delta: {:.5f}".format(delta))
            epoch += 1
        return V, values

    def improvement(self, V):
        tmp = np.dot(
            env.P_a.transpose(0, 2, 1),
            env.rewards + self.discount * V
        ).squeeze(2)

        #  This is looking for the maximum value across all the actions for
        #  every state. Then check if more than one action has that value
        #  and split the probability between the actions with the same max
        #  value
        max_val = tmp[np.arange(tmp.shape[0]), tmp.argmax(axis=1)][:, None]

        pi = (tmp == max_val).astype(int)
        pi = pi / pi.sum(axis=1)[:, None]
        for i, state in enumerate(self.policy.grid):
            if state.terminal:
                pi[i] = 0
        if (pi == self.policy.as_matrix()).all():
            return False
        else:
            self.policy.update(pi)
            return True


class ValueIteration(object):
    """docstring for ValueIteration"""

    def __init__(self, env, policy, discount, theta=0.0001):
        super(ValueIteration, self).__init__()
        self.env = env
        self.policy = policy
        self.theta = theta
        self.discount = discount

    def evaluation(self):
        # Make sure it starts higher than the stopping condition
        delta = self.theta * 10.
        V = np.zeros((len(policy), 1))
        pbar = tqdm(desc="delta {:.5f}".format(delta), leave=False)
        values = []
        n = self.policy.grid.n
        while delta > self.theta:
            _V = V.reshape((n, n))
            values.append((_V[-1, 0], _V[-1, -1]))
            v = V.copy()

            tmp = np.dot(
                env.P_a.transpose(0, 2, 1),
                env.rewards + self.discount * V
            ).squeeze(2)

            V = tmp.max(axis=1)[:, None]
            delta = np.abs(v - V).max()
            pbar.update(1)
            pbar.set_description("delta: {:.5f}".format(delta))
        return V, values

    def improvement(self, V):
        tmp = np.dot(
            env.P_a.transpose(0, 2, 1),
            env.rewards + self.discount * V
        ).squeeze(2)

        #  This is looking for the maximum value across all the actions for
        #  every state. Then check if more than one action has that value
        #  and split the probability between the actions with the same max
        #  value
        argmax = tmp.argmax(axis=1)
        max_val = tmp[np.arange(tmp.shape[0]), argmax][:, None]

        #  Mask terminal states. If the max value is zero across all actions
        #  then it's a terminal state.
        mask = (max_val != 0).astype(int)

        pi = np.zeros_like(tmp)
        pi[np.arange(pi.shape[0]), argmax] = 1.0
        pi *= mask
        self.policy.update(pi)


def policy_iteration(env, policy, discount, theta, k=float('inf')):
    improved = True
    history = {}
    i = 0
    while improved:
        iteration = PolicyIteration(env, policy, discount, theta, k)
        V, _V = iteration.evaluation()
        history[i] = _V
        n = policy.grid.n
        print(V.reshape((n, n)))
        improved = iteration.improvement(V)
        i += 1
    print(print_policy(policy))
    return history


def value_iteration(env, policy, discount, theta):
    iteration = PolicyIteration(env, policy, discount, theta)
    V, _V = iteration.evaluation()
    history = {0: _V}
    n = policy.grid.n
    print(V.reshape((n, n)))
    iteration.improvement(V)

    print(print_policy(policy))
    return history


def plot_history(iteration, history, prob, size):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6, 3))
    legend = []
    for epoch, values in history.items():
        left, right = zip(*values)
        legend.append(str(epoch))
        ax[0].plot(range(len(left)), left)
        ax[1].plot(range(len(right)), right)
    ax[0].set_title('Bottom left value')
    ax[1].set_title('Bottom right value')
    if iteration != 'value':
        ax[0].legend(legend)
        ax[1].legend(legend)
    fig.savefig(
        '{}_iteration_{}_{}.png'.format(
            iteration,
            str(int(prob * 100)),
            str(size)
        )
    )


if __name__ == '__main__':
    args = parse_args()
    grid = Grid(args.size)
    policy = Policy(grid)

    env = Environment(policy, args.prob)

    if args.iteration == 'policy':
        history = policy_iteration(env, policy, args.discount, args.theta)
        plot_history('policy', history, args.prob, args.size)
    elif args.iteration == 'value':
        history = value_iteration(env, policy, args.discount, args.theta)
        plot_history('value', history, args.prob, args.size)
    else:
        if args.k is None:
            raise Exception("Need to provide argument k for Modified Policy")
        history = policy_iteration(env, policy, args.discount, args.theta, args.k)
        plot_history('modified', history, args.prob, args.size)

    fp = "{}_{}_{}.json".format(args.iteration, int(100 * args.prob), args.size)
    with open(fp, 'w') as f:
        json.dump(history, f)
