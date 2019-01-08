#!/usr/bin/python2.7

"""Plots predicted pitch shifts versus ground-truth pitch shifts"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Matplotlib unavailable (probably on server). This program needs to be run locally.")
import numpy as np
import os

results_dir = "rnn_results"
epoch = 3
plot_training = True
plot_testing = False

startframe = 0
endframe = 1000


def plot_comparison(outputs, labels, title):
    num_tracks = labels.shape[1]
    for i in range(num_tracks):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
        ax1.plot(labels[:, i][startframe:endframe], color="blue")
        ax2.plot(outputs[:, i][startframe:endframe], color="red")
        plt.title(title)
        plt.show()

# training
if plot_training:
    for f in os.listdir(results_dir):
        if not f.endswith('.npy') or not "epoch_" + str(epoch) in f or not "training" in f:
            continue
        key = ('_').join(f.split('_')[-2:])[:-4]
        outputs = np.load(os.path.join(results_dir, "training_outputs_epoch_" + str(epoch) + "_" + key + ".npy"))
        labels = np.load(os.path.join(results_dir, "training_labels_epoch_" + str(epoch) + "_" + key + ".npy"))
        plot_comparison(outputs, labels, "Training")

# testing
if plot_testing:
    for f in os.listdir(results_dir):
        if not f.endswith('.npy') or not "epoch_" + str(epoch) in f or not "testing" in f:
            continue
        key = ('_').join(f.split('_')[-2:])[:-4]
        outputs = np.load(os.path.join(results_dir, "testing_outputs_epoch_" + str(epoch) + "_" + key + ".npy"))
        labels = np.load(os.path.join(results_dir, "testing_labels_epoch_" + str(epoch) + "_" + key + ".npy"))
        plot_comparison(outputs, labels, "Testing")
