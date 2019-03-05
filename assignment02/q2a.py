import numpy as np
import gym
import argparse
from pathlib import Path
import os
import tqdm
from rlai import IHT, tiles


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        help="Number of independant runs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--episodes",
        help=(
            "Number of segments. Each one will start at (0, 0)"
        ),
        type=int,
        default=200
    )

    parser.add_argument(
        "--torque_prob",
        help=(
            "produces torque in the same direction as the current velocity "
            "with probability p and in the opposite direction with "
            "probability (1-p). If velocity is 0, you can torque in a "
            "random direction."
        ),
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--tiling",
        help=(
            "Number of overlapping tiling for the discretization of "
            "the angular position and angular velocity"
            "NOTE: The number of overlapping tilings will divide the "
            "learning rate."
        ),
        type=int,
        default=5
    )

    parser.add_argument(
        "--lambda",
        help="Discount factor",
        default=[0, 0.3, 0.7, 0.9, 1],
        type=float,
        nargs="*"
    )

    parser.add_argument(
        "--alpha",
        help=(
            "Learning rate"
            "NOTE: The learning rate will be divided by the number of "
            "overlapping tilings."
        ),
        type=float,
        default=[1 / 4, 1 / 8, 1 / 16],
        nargs="*"
    )

    parser.add_argument(
        "-s", "--save",
        help="Path where to save all the files",
        default=Path('.'),
        type=Path
    )

    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    return args


class Agent(object):
    """
    Agent running in the environment `Taxi-V2` from OpenAI.
    """

    def __init__(
        self,
        method,
        temperature=1.0,
        max_steps=100,
        alpha=0.1,
        gamma=0.9,
        lamb=0.9
    ):
        super(Agent, self).__init__()
        if method not in ['sarsa', 'expected_sarsa', 'q_learning']:
            raise ValueError("Method not supported")

        self.method = method
        self.temperature = temperature
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.env = gym.make("Pendulum-v0")
        num_actions = self.env.action_space.n

        # THESE COME FROM THE TILE-CODING (10x10 grid!)
        self.eligibility = np.zeros(N_STATES)
        self.q_table = np.zeros(
            (self.env.observation_space.n, num_actions)
        )

    def run_episode(self):
        # Initial state for the episode. State is a number, so we can
        # use it to index our q_table and policy
        episode_reward = 0
        episode_errors = []
        done = False

        state = self.env.reset()

        # At worst, will terminate at env._max_episode_steps.
        while not done:

            # Take action according to our policy.
            action = softmax(self.q_table[state], self.temperature)

            s_prime, reward, done, _ = self.env.step(action)

            self.eligibility *= lamb * gamma
            self.eligibility[state] += 1.0

            # get the td-error and update every state's value estimate
            # according to their eligibilities.
            error = reward + self.gamma * self.q_table[new_state] - self.q_table[state]
            self.state_values = self.q_table + self.alpha * error * self.eligibility

            # Set t+1 to t, for the next loop.
            state = s_prime

        rms_error = np.sqrt(np.mean( np.array(episode_errors)**2 ))

        return(rms_error, reward)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    seeds = list(range(42, 42 + args.runs))

