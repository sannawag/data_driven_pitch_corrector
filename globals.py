#!/usr/bin/python3.6


"""Directories and audio parameters used globally"""

import os

# audio parameters
global_fs = 22050  # assumed fs for all vocal tracks
framesPerSec = 86  # corresponds to a hopSize of 256 samples when fs = 22050
frameSize = 2048
hopSize = 256

# analysis parameters
restrict_range = False  # restrict audio analysis from start_sec to end_sec for whole pipeline
start_sec = 30  # range of file when running analysis on only a part of it
end_sec = 90
pitch_shift_versions = 7  # number of pitch-shifted versions per performance

# shifts
max_semitone = 1  # max shift amount when de-tuning

# CQT parameters
cqt_params = dict()
cqt_params['fmin'] = 125  # lowest measured frequency
cqt_params['num_8va'] = 5.5  # total number of octaves
cqt_params['bins_per_note'] = 16
cqt_params['bins_per_8va'] = 12 * cqt_params['bins_per_note']
cqt_params['total_bins'] = int(cqt_params['num_8va'] * cqt_params['bins_per_8va'])
cqt_params['normalizing_constant'] = 0.01  # used for audio normalization
cqt_params['feature_size'] = cqt_params['total_bins'] * 2  # data feature size as input to ML model


# directories
base_directory = os.environ.get("INTONATION", "./Intonation")
print("Data root directory is set to", base_directory)
plot_directory = "./plots"
parameter_plot_directory = os.path.join(plot_directory, "parameter_visualization")
layer_plot_directory = os.path.join(plot_directory, "layer_visualization")
pytorch_models_directory = "./pytorch_checkpoints_and_models"

# training data
pyin_directory = os.path.join(base_directory, "training_data/pyin")
raw_audio_directory = os.path.join(base_directory, "training_data/raw_audio")
vocals_directory = os.path.join(raw_audio_directory, "vocal_tracks")
backing_tracks_directory = os.path.join(raw_audio_directory, "backing_tracks_wav")

# validation data
realworld_pyin_directory = os.path.join(base_directory, "realworld_data/pyin")
realworld_raw_audio_directory = os.path.join(base_directory, "realworld_data/raw_audio")
realworld_vocals_directory = os.path.join(realworld_raw_audio_directory, "vocal_tracks")
realworld_backing_tracks_directory = os.path.join(realworld_raw_audio_directory, "backing_tracks_wav")

# csv files listing performance ids and arrangement ids
metadata_csv = os.path.join(base_directory, "intonation.csv")  # training
realworld_csv = os.path.join(base_directory, "realworld.csv")  # autotuning
