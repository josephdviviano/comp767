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

    def run_episode(self, greedy=False):
        #  TODO ensure that the starting and end location of the passenger
        #  are random

        # Initial state for the episode. State is a number, so we can
        # use it to index our q_table and policy
        episode_reward = 0
        episode_errors = []
        done = False

        state = self.env.reset()

        # At worst, will terminate at env._max_episode_steps.
        while not done:

            if greedy:
                action = np.argmax(self.q_table[state])
            else:
                # Take action according to our policy.
                action = softmax(self.q_table[state], self.temperature)


            s_prime, reward, done, _ = self.env.step(action)

            if self.method == 'q_learning':
                error = self.q_learning_error(s_prime, state, reward, action)

            elif self.method == 'sarsa':
                error = self.sarsa_error(s_prime, state, reward, action)

            elif self.method == 'expected_sarsa':
                error = self.expected_sarsa_error(s_prime, state, reward, action)

            episode_reward += reward
            episode_errors.append(error)

            # Don't update policy during greedy runs.
            if not greedy:
                # Update estimate of Q(s, a) using a small step size on error.
                self.q_table[state, action] = self.q_table[state, action] + self.alpha * error

            # Set t+1 to t, for the next loop.
            state = s_prime

        rms_error = np.sqrt(np.mean( np.array(episode_errors)**2 ))

        return(rms_error, reward)

    def q_learning_error(self, s_prime, state, reward, action):
        """Q learning takes the max action from state s_prime."""
        # Tiebreak actions by taking the first.
        a_prime = self.q[s_prime].argmax()

        error = reward
        error += self.gamma * self.q_table[s_prime, a_prime]
        error -= self.q_table[state, action]

        return(error)

    def sarsa_error(self, s_prime, state, reward, action):
        """Sarsa samples again from the policy to get a_prime."""
        a_prime = softmax(self.q_table[s_prime], self.temperature)

        error = reward
        error += self.gamma * self.q_table[s_prime, a_prime]
        error -= self.q_table[state, action]

        return(error)

    def expected_sarsa_error(self, s_prime, state, reward, action):
        """
        Takes a weighted average of all possible action from state s_prime.
        """
        probs = softmax(self.q_table[s_prime], self.temperature, sample=False)
        expectation = (probs * self.q_table[s_prime]).sum()

        error = reward
        error += self.gamma * expectation
        error -= self.q_table[state, action]

        return(error)


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

    alphas = np.array([0.01, 0.5, 0.9])
    temps = np.array([0.1, 1.0, 10])

    errors = np.zeros((len(alphas), len(temps), args.segments, args.runs))
    rewards = np.zeros_like(errors)

    for i, alpha in enumerate(tqdm.tqdm(alphas, desc="alpha")):
        for j, temp in enumerate(tqdm.tqdm(temps, desc="temp")):
            for run in tqdm.trange(args.runs, desc="Run"):

                agent = Agent(
                    method=args.method,
                    alpha=alpha,
                    temperature=temp,
                    gamma=args.gamma
                )

                for segment in tqdm.trange(args.segments, desc="Segment"):

                    for episode in range(N_EPISODES):
                        _, _ = agent.run_episode()

                    error, reward = agent.run_episode(greedy=True)
                    errors[i, j, segment, run] = error
                    rewards[i, j, segment, run] = reward

    np.save(args.save / "errors.npy", errors)
    np.save(args.save / "rewards.npy", rewards)



