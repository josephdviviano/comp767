#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

SUBSET = 'rewards'

class RunStats():
    def __init__(self, x, name):
        self.data = x[name]
        self.name = name

    def seg_mean(self):
        return({
            'rewards': np.mean(self.data['rewards'], axis=3),
            'errors': np.mean(self.data['errors'], axis=3)})

    def seg_std(self):
        return({
            'rewards': np.std(self.data['rewards'], axis=3),
            'errors': np.std(self.data['errors'], axis=3)})


def plot_hp_comparisons(split):

    LEGEND = ['Temp=0.1', 'Temp=1.0', 'Temp=10']
    XTICKLABELS = [0.01, 0.5, 0.9]
    XTICKS = [0, 1, 2]
    YLIM = [-3, 3]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(np.mean(split.seg_mean()[SUBSET][0, :, :, :], axis=2))
    ax1.set_ylabel('Return')
    ax1.set_xticks(XTICKS)
    ax1.set_xticklabels([])
    ax1.set_ylim(YLIM)
    ax1.legend(LEGEND)

    ax2.plot(np.mean(split.seg_mean()[SUBSET][1, :, :, :], axis=2))
    ax2.set_ylabel('Return')
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([])
    ax2.set_ylim(YLIM)
    ax2.legend(LEGEND)

    ax3.plot(np.mean(split.seg_mean()[SUBSET][2, :, :, :], axis=2))
    ax3.set_ylabel('Return')
    ax3.set_xlabel('Alpha')
    ax3.set_xticks(XTICKS)
    ax3.set_xticklabels(XTICKLABELS)
    ax3.set_ylim(YLIM)
    ax3.legend(LEGEND)

    f.savefig('img/{}_alpha_temp_return.jpg'.format(split.name))


def plot_learning_curves(train, test):

    LEGEND = ['train', 'test']
    XTICKLABELS = np.arange(10) + 1
    XTICKS = range(10)
    YLIM = [-5, 5]
    ALPHA = 1
    TEMP = 1

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(train.seg_mean()[SUBSET][0, ALPHA, TEMP, :])
    ax1.plot(test.seg_mean()[SUBSET][0, ALPHA, TEMP, :])
    ax1.set_ylabel('Return')
    ax1.set_xticks(XTICKS)
    ax1.set_xticklabels([])
    ax1.set_ylim(YLIM)
    ax1.legend(LEGEND)

    ax2.plot(train.seg_mean()[SUBSET][1, ALPHA, TEMP, :])
    ax2.plot(test.seg_mean()[SUBSET][1, ALPHA, TEMP, :])
    ax2.set_ylabel('Return')
    ax2.set_xticks(XTICKS)
    ax2.set_xticklabels([])
    ax2.set_ylim(YLIM)
    ax2.legend(LEGEND)

    ax3.plot(train.seg_mean()[SUBSET][2, ALPHA, TEMP, :])
    ax3.plot(test.seg_mean()[SUBSET][2, ALPHA, TEMP, :])
    ax3.set_ylabel('Return')
    ax3.set_xlabel('Run')
    ax3.set_xticks(XTICKS)
    ax3.set_xticklabels(XTICKLABELS)
    ax3.set_ylim(YLIM)
    ax3.legend(LEGEND)

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

