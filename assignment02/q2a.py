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


if __name__ == '__main__':
    args = parse_args()
    print(args)
    seeds = list(range(42, 42 + args.runs))
