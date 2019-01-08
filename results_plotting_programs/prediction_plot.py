
"""Plots de-tuned, corrected, and ground-truth pitch curves"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import matplotlib.ticker as ticker

model = "51"
base_dir = "/Users/scwager/Downloads/autotune_temp/results" + model + "/rnn_results"

id = "12345"  # substitute ID of selected performance
epoch = "13"
pkl_fpath = os.path.join(base_dir, id + ".pkl")
out_fpath = os.path.join(base_dir, "testing_epoch_" + epoch + "_outputs_" + id + ".npy")
gt_fpath = os.path.join(base_dir, "testing_epoch_" + epoch + "_labels_" + id + ".npy")

with open(pkl_fpath, "rb") as f:
    data_dict = pkl.load(f)
outputs = np.load(out_fpath)
labels = np.load(gt_fpath)

midi = data_dict['midi']
notes = data_dict['notes']
stacked_cqt_v = data_dict['spect_v']
cqt_b = data_dict['spect_b']
stacked_cqt_combined = data_dict['spect_c']
shifted_pyin = data_dict['shifted_pyin']
note_shifts = data_dict['shifts_gt']
original_boundaries = data_dict['original_boundaries']

# load the original, in-tune pitch track
pitch_track = np.load(os.path.join(base_dir, "pyin_" + id + ".npy"))
frames = len(pitch_track)

# convert to shape < notes, shifts >
outputs = np.squeeze(outputs)
labels = np.squeeze(labels)

# plot the shifts after applying them to the original frame indices
frame_outputs_0 = np.zeros(frames)
frame_labels_0 = np.zeros(frames)
frame_outputs_5 = np.zeros(frames)
frame_labels_5 = np.zeros(frames)

idx = 2

for i in range(len(labels[:, 0])):
    frame_outputs_5[original_boundaries[i, 0]: original_boundaries[i, 1]] += outputs[i, idx]
    frame_labels_5[original_boundaries[i, 0]: original_boundaries[i, 1]] += labels[i, idx]

# shift pitch to get the de-tuned input to the neural net, then apply correction (negative of learned shift)
shifted_pitch_track_0 = np.copy(pitch_track)
corrected_pitch_track_0 = np.copy(pitch_track)
shifted_pitch_track_5 = np.copy(pitch_track)
corrected_pitch_track_5 = np.copy(pitch_track)
for i in range(len(labels[:, 0])):
    shifted_pitch_track_5[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                np.power(2, -labels[i, idx] / 12.0)
    corrected_pitch_track_5[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                np.power(2, (-labels[i, idx] + outputs[i, idx]) / 12.0)

shifted_pitch_track_5[shifted_pitch_track_5 < 1] /= 0
corrected_pitch_track_5[corrected_pitch_track_5 < 1] /= 0

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
ax[0].grid(True)
ax[1].grid(True)
ax[0].plot(frame_labels_5[1900:4100], color="orange", label="ground truth")
ax[0].plot(frame_outputs_5[1900:4100], color="blue", linestyle="dashed", label="predicted shift")
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.10))
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x * 100)))
ax[0].yaxis.set_major_formatter(ticks_y)
ax[0].set_ylabel("Cents")
ax[0].legend(fontsize=13)
ax[0].xaxis.label.set_size(13)
ax[0].yaxis.label.set_size(13)
ax[1].set_ylim(np.log2(390), np.log2(720))
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(np.power(2, x))))
ax[1].yaxis.set_major_formatter(ticks_y)
ax[1].yaxis.set_major_locator(plt.FixedLocator(np.log2(np.arange(50, 1000, 50))))
ax[1].plot(np.log2(corrected_pitch_track_5[1900:4100]), linewidth=2, linestyle="solid", color="green",
           label="autotuned")
ax[1].plot(np.log2(pitch_track[1900:4100]), linewidth=1, linestyle="solid", color="black", label="original")
ax[1].plot(np.log2(shifted_pitch_track_5[1900:4100]), linewidth=0.5, linestyle="solid", color="red", label="de-tuned")
ax[1].xaxis.label.set_size(13)
ax[1].yaxis.label.set_size(13)
ax[1].set_xlabel("Frames")
ax[1].set_ylabel("Frequency (Hz)")
ax[1].legend(fontsize=13)
plt.tight_layout()
plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/results" + model + ".eps", format="eps",
            bbox_inches='tight')
plt.show()
