#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(data):

    LEGEND = ['Alpha=1/4', 'Alpha=1/8', 'Alpha=1/16']
    XTICKS = np.arange(0, 201, 20)
    XTICKLABELS = XTICKS
    XTICKLABELS[0] = 1
    YLIM = [-75, 20]
    YLABEL = 'Value at (0,0), Goal State'
    TITLES = ['Lambda=0.0',
              'Lambda=0.3',
              'Lambda=0.7',
              'Lambda=0.9',
              'Lambda=1.0']

    f, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 12))

    for i, ax in enumerate(axs):

        ax.plot(np.mean(data[:, :, i, :], axis=0))
        ax.set_ylabel(YLABEL)
        ax.set_xticks(XTICKS)
        ax.set_ylim(YLIM)
        ax.legend(LEGEND)
        ax.set_title(TITLES[i])

        if i+1 == len(axs):
            ax.set_xticklabels(XTICKLABELS)
            ax.set_xlabel('Episode')
        else:
            ax.set_xticklabels([])

    plt.tight_layout()

    f.savefig('img/starting_values.jpg')


def main():

    data = np.load('data/q2a.npy')
    plot_learning_curves(data)


if __name__ == "__main__":
    main()
