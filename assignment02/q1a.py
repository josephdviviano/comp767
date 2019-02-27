

import numpy as np
import gym


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


def softmax(action_values, temperature=1.0, sample=True):
    exp_val = np.exp(action_values) / temperature
    probs = exp_val / exp_val.sum()
    if sample:
        return np.where(probs.cumsum() > np.random.rand())[0][0]
    else:
        return probs
