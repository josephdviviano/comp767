

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


State = namedtuple('State', ['i', 'j', 'reward'])


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
    out += "\n└" + "───┴" * (n - 1) + "───┘"
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
                    state = State(i, j, 10)
                elif i == 0 and j == (n - 1):
                    state = State(i, j, 1)
                else:
                    state = State(i, j, 0)
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
        self._policy = {
            s: {
                Action.UP: 0.25,
                Action.RIGHT: 0.25,
                Action.DOWN: 0.25,
                Action.LEFT: 0.25
            } for s in grid
        }

    def __getitem__(self, state):
        return self._policy[state]

    def __len__(self):
        return len(self._policy)


class Transition(object):
    """docstring for Transition"""

    def __init__(self, policy, p):
        super(Transition, self).__init__()
        self.policy = policy
        self._grid = policy.grid
        self._state_idx = {state: i for i, state in enumerate(policy.grid)}
        self.p = p

    @property
    def matrix_tensor(self):
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
                        t[i, j, k] += (1 - self.p) / (num_actions - 1) * action_prob
        return t

    @property
    def matrix(self):
        return self.matrix_tensor.sum(axis=2)
