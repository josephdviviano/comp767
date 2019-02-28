#!/usr/bin/env python

import numpy as np
import gym
import argparse
from pathlib import Path
import os
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        help="Number of independant runs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--segments",
        help=(
            "Number of segments. Each segment there are 10 episodes "
            "of training, followed by 1 episode in which you simply "
            "run the optimal policy so far."
        ),
        type=int,
        default=100
    )

    parser.add_argument(
        "--steps",
        help=(
            "Maximum number of steps in an episode."
        ),
        type=int,
        default=100
    )

    parser.add_argument(
        "--temperature",
        help=(
            "Temperature value that will control the exploration. "
            "Higher value will increase exploration. Lower value "
            "will increase exploitation."
        ),
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--alpha",
        help="Learning rate",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--gamma",
        help="Discount factor",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "-s", "--save",
        help="Path where to save all the files",
        default=Path('.'),
        type=Path
    )

    parser.add_argument(
        "-m", "--method",
        help="Method to use",
        default="sarsa",
        choices=["sarsa", "expected_sarsa", "q_learning"]
    )

    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("Temperature needs to be greater than zero.")
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
        gamma=0.9
    ):
        super(Agent, self).__init__()
        if method not in ['sarsa', 'expected_sarsa', 'q_learning']:
            raise ValueError("Method not supported")

        self.method = method
        self.temperature = temperature
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make("Taxi-v2")
        num_actions = self.env.action_space.n
        self.q_table = np.zeros(
            (self.env.observation_space.n, num_actions)
        )

    def run_episode(self):
        #  TODO ensure that the starting and end location of the passenger
        #  are random

        # Initial state for the episode. State is a number, so we can
        # use it to index our q_table and policy
        state = self.env.reset()
        done = False
        step = 0
        while not done:
            # take action according to our policy
            action = softmax(self.q_table[state], self.temperature)
            # Take action in the environment
            s_prime, reward, done, _ = self.env.step(action)

            if self.method == 'q_learning':
                # Q learning takes the max action from state s_prime
                # We can potentially have multiple actions with the same
                # max values. But no indication is given as to how to break
                # equality so we take the first one returned by argmax
                a_prime = self.q[s_prime].argmax()

                # update the q_table
                # First calculate the error
                error = reward
                error += self.gamma * self.q_table[s_prime, a_prime]
                error -= self.q_table[state, action]
                # Then update the current estimate of Q(s, a) by updating
                # in the direction of the error with a small step
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * error

            elif self.method == 'sarsa':
                # Sarsa samples again from the policy to get action_prime
                a_prime = softmax(self.q_table[s_prime], self.temperature)

                # update the q_table
                # First calculate the error
                error = reward
                error += self.gamma * self.q_table[s_prime, a_prime]
                error -= self.q_table[state, action]
                # Then update the current estimate of Q(s, a) by updating
                # in the direction of the error with a small step
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * error

            elif self.method == 'expected_sarsa':
                # Expected Sarsa takes a weighted average of all possible
                # action from state s_prime
                probs = softmax(
                    self.q_table[s_prime],
                    self.temperature,
                    sample=False
                )
                expectation = (probs * self.q_table[s_prime]).sum()

                # update the q_table
                # First calculate the error
                error = reward
                error += self.gamma * expectation
                error -= self.q_table[state, action]
                # Then update the current estimate of Q(s, a) by updating
                # in the direction of the error with a small step
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * error

            step += 1
            if step >= self.max_steps:
                done = True

            state = s_prime

    def run_greedy_episode(self):
        step = 0
        reward = 0

        state = self.env.reset()
        done = False
        step = 0

        while not done:
            action = np.argmax(self.q_table[state])
            state, r, done, _ = self.env.step(action)
            reward += r
            step += 1
            if step == self.max_steps:
                done = True
        return step, reward


def softmax(action_values, temperature=1.0, sample=True):
    exp_val = np.exp(action_values) / temperature
    probs = exp_val / exp_val.sum()
    if sample:
        return np.where(probs.cumsum() > np.random.rand())[0][0]
    else:
        return probs


if __name__ == "__main__":
    args = parse_args()

    N_EPISODES = 10

    alphas = np.linspace(0.01, 0.5, 1)
    temps = np.array([0.1, 1.0, 10])

    steps = np.zeros((len(alphas), len(temps), args.segments, args.runs))
    rewards = np.zeros_like(steps)

    for i, alpha in enumerate(tqdm.tqdm(alphas, desc="alpha")):
        for j, temp in enumerate(tqdm.tqdm(temps, desc="temp")):
            for run in tqdm.trange(args.runs, desc="Run"):

                agent = Agent(
                    method=args.method,
                    alpha=alpha,
                    temperature=temp,
                    max_steps=args.steps,
                    gamma=args.gamma
                )

                for segment in tqdm.trange(args.segments, desc="Segment"):

                    for episode in range(N_EPISODES):
                        agent.run_episode()

                    step, reward = agent.run_greedy_episode()
                    steps[i, j, segment, run] = step
                    rewards[i, j, segment, run] = reward

    np.save(args.save / "steps.npy", steps)
    np.save(args.save / "rewards.npy", rewards)



