#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

SUBSET = 'rewards'

class RunStats():
    def __init__(self, x, name):
        self.data = x[name]
        self.name = name

    def best_seg(self):
        return(self.data[:, :, :, -1, :])

    def run_stats(self):
        return([np.mean(self.data, axis=4), np.std(self.data, axis=4)])


def plot_hp_comparisons(split):

    LEGEND = ['Temp=0.1', 'Temp=1.0', 'Temp=10']
    XTICKLABELS = [0.1, 0.3, 0.5, 0.7, 0.9]
    XTICKS = range(5)
    X = range(5)
    YLIM = [-500, 50]
    TITLES = ['Sarsa', 'Expected Sarsa', 'Q Learning']

    f, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    for i, ax in enumerate(axs):

        ax.plot(np.mean(split.best_seg()[i, :, :, :], axis=2))
        ax.set_ylabel('Return')
        ax.set_xticks(XTICKS)
        ax.set_ylim(YLIM)
        ax.legend(LEGEND)
        ax.set_title(TITLES[i])

        if i+1 == len(axs):
            ax.set_xticklabels(XTICKLABELS)
            ax.set_xlabel('Alpha')
        else:
            ax.set_xticklabels([])

    plt.tight_layout()

    f.savefig('img/{}_alpha_temp_return.jpg'.format(split.name))


def plot_learning_curves(train, test):

    LEGEND = ['train', 'test']
    X = np.arange(100)
    XTICKS = np.arange(0, 101, 10)
    XTICKLABELS = XTICKS
    XTICKLABELS[0] = 1
    YLIM = [-1000, 200]
    ALPHA = 2
    TEMP = 1
    TITLES = ['Sarsa', 'Expected Sarsa', 'Q Learning']

    f, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    for i, ax in enumerate(axs):
        ax.errorbar(X,
                     train.run_stats()[0][i, ALPHA, TEMP, :],
                     train.run_stats()[1][i, ALPHA, TEMP, :])
        ax.errorbar(X,
                     test.run_stats()[0][i, ALPHA, TEMP, :],
                     test.run_stats()[1][i, ALPHA, TEMP, :])
        ax.set_ylabel('Return')
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([])
        ax.set_ylim(YLIM)
        ax.legend(LEGEND)
        ax.set_title(TITLES[i])

        if i+1 == len(axs):
            ax.set_xticklabels(XTICKLABELS)
            ax.set_xlabel('Episode')
        else:
            ax.set_xticklabels([])

    plt.tight_layout()

    f.savefig('img/learning_curves.jpg')


def main():

    data = np.load('data/q1a.npy')
    train = RunStats(data.item(), 'train')
    test = RunStats(data.item(), 'test')

    plot_hp_comparisons(train)
    plot_hp_comparisons(test)

    plot_learning_curves(train, test)


if __name__ == "__main__":
    main()
