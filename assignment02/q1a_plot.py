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

    # Stores best hyperparameters for each experiment
    # [(Alpha1, Temp1), (Alpha2, Temp2), (Alpha3, Temp3)].
    hyperparameters = []

    f, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    for i, ax in enumerate(axs):

        results = np.mean(split.best_seg()[i, :, :, :], axis=2)

        # Get hyperparametes for maximum return.
        alpha, temp = np.where(results == np.max(results))
        hyperparameters.append([alpha[0], temp[0]])

        ax.plot(results)
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

    return(hyperparameters)


def plot_learning_curves(train, test, hyperparameters):

    LEGEND = ['train', 'test']
    X = np.arange(100)
    XTICKS = np.arange(0, 101, 10)
    XTICKLABELS = np.arange(0, 101, 10)
    XTICKLABELS[0] = 1
    YLIM = [-1000, 200]
    ALPHAS = ['0.1', '0.3', '0.5', '0.7', '0.9']
    TEMPS = ['0.1', '1.0', '10']
    TITLES = ['Sarsa', 'Expected Sarsa', 'Q Learning']

    f, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 12))

    for i, ax in enumerate(axs):

        alpha, temp = hyperparameters[i]

        ax.errorbar(X,
                     train.run_stats()[0][i, alpha, temp, :],
                     train.run_stats()[1][i, alpha, temp, :])
        ax.errorbar(X,
                     test.run_stats()[0][i, alpha, temp, :],
                     test.run_stats()[1][i, alpha, temp, :])
        ax.set_ylabel('Return')
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([])
        ax.set_ylim(YLIM)
        ax.legend(LEGEND)

        title = "{}: alpha={}, temp={}".format(
            TITLES[i], ALPHAS[alpha], TEMPS[temp])

        ax.set_title(title)

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

    _ = plot_hp_comparisons(train)
    hyperparameters = plot_hp_comparisons(test)

    plot_learning_curves(train, test, hyperparameters)


if __name__ == "__main__":
    main()
