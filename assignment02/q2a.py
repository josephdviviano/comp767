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
        "--trace_decay",
        help="Trace decay",
        default=[0, 0.3, 0.7, 0.9, 1],
        type=float,
        nargs="*"
    )

    parser.add_argument(
        "--gamma",
        help="Discount factor",
        default=1.0,
        type=float
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
        alpha=0.1,
        gamma=0.9,
        trace_decay=0.9,
        torque_prob=0.9,
        tilings=5,
        seed=None
    ):
        super(Agent, self).__init__()
        np.random.seed(seed)

        self.alpha = alpha
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.tilings = tilings
        self.env = gym.make("Pendulum-v0")

        # THESE COME FROM THE TILE-CODING (10x10 grid!)
        self.eligibility = np.zeros(self.tilings * 10 * 10)
        self.weights = np.random.uniform(
            -0.001,
            0.001,
            size=self.tilings * 10 * 10
        )

    def run_episode(self):
        def mytiles(angle, velocity):
            # Rescale angle from (-1, 1) to (0, 10)
            # Rescale velocity from (-8, 8) to (0, 10)
            angle_scale_factor = 10.0 / 2
            velocity_scale_factor = 10.0 / 16
            return tiles(
                iht,
                self.tilings,
                [
                    (angle + 1) * angle_scale_factor,
                    (velocity + 8) * velocity_scale_factor
                ]
            )

        def vec_from_tiles(tiles):
            x = np.zeros(self.tilings * 10 * 10)
            for idx in tiles:
                x[idx] = 1
            return x

        iht = IHT(self.tilings * 10 * 10)
        eligibility = np.zeros(self.tilings * 10 * 10)

        episode_reward = 0
        episode_errors = []
        done = False

        # state[0] = cos(theta), ranges from -1 to 1
        # state[1] = sin(theta), ranges from -1 to 1
        # state[2] = angular velocity , ranges from -8 to 8
        state = self.env.reset()

        # At worst, will terminate at env._max_episode_steps.
        while not done:
            # This is the input to the value function
            x = vec_from_tiles(mytiles(state[0], state[2]))

            # evaluate the fixed policy that produces torque in the same
            # direction as the current velocity with probability 0.9 and
            # in the opposite direction with probability 0.1
            # If velocity is 0, you can torque in a random direction.
            velocity_direction = np.sign(state[2])
            rand = np.random.rand()
            if (state[2] == 0 and rand < 0.5) or rand > 0.9:
                velocity_direction *= -1

            # TODO figure out how to take an action. For now randomly
            # Action range from -2 to 2. Sample from 0 to 2 and multiply
            # by the velocity_direction. This is apply torque in the direction
            # asked in the question
            action = np.random.uniform(0, 2) * velocity_direction

            # Take action according to our policy.
            s_prime, reward, done, _ = self.env.step([action])
            episode_reward += reward

            # This returns a list of {self.tilings} integers. Those are the
            # states we want to update

            # ∇v(S,w) = x
            x_prime = vec_from_tiles(mytiles(s_prime[0], s_prime[2]))

            eligibility *= self.trace_decay * self.gamma
            eligibility += x

            delta = reward + self.gamma * np.dot(self.weights, x_prime)
            delta -= np.dot(self.weights, x)
            episode_errors.append(delta)

            # Update the weights
            self.weights += self.alpha * delta * eligibility

            # Set t+1 to t, for the next loop.
            state = s_prime

        rms_error = np.sqrt(np.mean(np.array(episode_errors)**2))

        return(rms_error, episode_reward)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    seeds = list(range(42, 42 + args.runs))
    for run in tqdm.trange(args.runs, desc='Run'):
        for trace_decay in args.trace_decay:
            for alpha in args.alpha:
                agent = Agent(
                    alpha=alpha,
                    gamma=args.gamma,
                    trace_decay=trace_decay,
                    torque_prob=args.torque_prob,
                    tilings=args.tiling,
                    seed=seeds[run]
                )
                agent.run_episode()
