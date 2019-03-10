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

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    ax1.plot(np.mean(split.best_seg()[0, :, :, :], axis=2))
    ax1.set_ylabel('Return')
    ax1.set_xticks(XTICKS)
    ax1.set_xticklabels([])
    ax1.set_ylim(YLIM)
    ax1.legend(LEGEND)
    ax1.set_title('Sarsa')

    ax2.plot(np.mean(split.best_seg()[1, :, :, :], axis=2))
    ax2.set_ylabel('Return')
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([])
    ax2.set_ylim(YLIM)
    ax2.legend(LEGEND)
    ax2.set_title('Expected Sarsa')

    ax3.plot(np.mean(split.best_seg()[2, :, :, :], axis=2))
    ax3.set_ylabel('Return')
    ax3.set_xlabel('Alpha')
    ax3.set_xticks(XTICKS)
    ax3.set_xticklabels(XTICKLABELS)
    ax3.set_ylim(YLIM)
    ax3.legend(LEGEND)
    ax3.set_title('Q Learning')

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

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 12))
    ax1.errorbar(X, train.run_stats()[0][0, ALPHA, TEMP, :], train.run_stats()[1][0, ALPHA, TEMP, :])
    ax1.errorbar(X, test.run_stats()[0][0, ALPHA, TEMP, :], test.run_stats()[1][0, ALPHA, TEMP, :])
    ax1.set_ylabel('Return')
    ax1.set_xticks(XTICKS)
    ax1.set_xticklabels([])
    ax1.set_ylim(YLIM)
    ax1.legend(LEGEND)
    ax1.set_title('Sarsa')

    ax2.errorbar(X, train.run_stats()[0][1, ALPHA, TEMP, :], train.run_stats()[1][1, ALPHA, TEMP, :])
    ax2.errorbar(X, test.run_stats()[0][1, ALPHA, TEMP, :], test.run_stats()[1][1, ALPHA, TEMP, :])
    ax2.set_ylabel('Return')
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([])
    ax2.set_ylim(YLIM)
    ax2.legend(LEGEND)
    ax2.set_title('Expected Sarsa')

    ax3.errorbar(X, train.run_stats()[0][2, ALPHA, TEMP, :], train.run_stats()[1][2, ALPHA, TEMP, :])
    ax3.errorbar(X, test.run_stats()[0][2, ALPHA, TEMP, :], test.run_stats()[1][2, ALPHA, TEMP, :])
    ax3.set_ylabel('Return')
    ax3.set_xlabel('Run')
    ax3.set_xticks(XTICKS)
    ax3.set_xticklabels(XTICKLABELS)
    ax3.set_ylim(YLIM)
    ax3.legend(LEGEND)
    ax3.set_title('Q Learning')

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

