#!/usr/bin/python3.6


"""Directories and audio parameters used globally"""


from __future__ import print_function

__version__ = 2

import os

# audio parameters
global_fs = 22050  # assumed fs for all vocal tracks
framesPerSec = 86  # corresponds to a hopSize of 256 samples when fs = 22050
frameSize = 2048
hopSize = 256

# analysis parameters
restrict_range = True  # restrict audio analysis from start_sec to end_sec for whole pipeline
start_sec = 30  # range of file when running analysis on only a part of it
end_sec = 90
pitch_shift_versions = 5  # number of pitch-shifted versions per performance

# shifts
max_semitone = 1  # max shift amount when de-tuning

# CQT parameters
cqt_params = dict()
cqt_params['fmin'] = 100  # lowest measured frequency
cqt_params['num_8va'] = 6  # total number of octaves
cqt_params['bins_per_8va'] = 12 * 8  # bins per octave (in cents: divide by 12 to get bins per note)
cqt_params['total_bins'] = cqt_params['num_8va'] * cqt_params['bins_per_8va']
cqt_params['normalizing_constant'] = 0.01  # used for audio normalization
cqt_params['feature_size'] = cqt_params['total_bins'] * 2  # data feature size as input to ML model

# chroma
chroma_dim = 12

# directories
base_directory = os.environ.get("INTONATION", "./Intonation")
print("Data directory is set to", base_directory)
shift_ground_truth_directory = os.path.join(base_directory, "frame_shifts")
solo_cqt_directory = os.path.join(base_directory, "solo_cqt")
back_cqt_directory = os.path.join(base_directory, "back_cqt")
back_chroma_directory = os.path.join(base_directory, "back_chroma")
pyin_directory = os.path.join(base_directory, "vocals_pitch_pyin")
midi_directory = os.path.join(base_directory, "vocals_midi")

raw_audio_directory = os.path.join(base_directory, "raw_audio/")
audio_sample_directory = os.path.join(raw_audio_directory, "audio_corresponding_to_pitch_track")
audio_shifted_directory = os.path.join(raw_audio_directory, "pitch_shifted_audio")
audio_resynthesized_directory = os.path.join(raw_audio_directory, "resynthesized_audio")
vocals_directory = os.path.join(raw_audio_directory, "vocal_tracks")
backing_tracks_directory = os.path.join(raw_audio_directory, "backing_tracks_wav")
plot_directory = "./plots"
parameter_plot_directory = os.path.join(plot_directory, "parameter_visualization")
layer_plot_directory = os.path.join(plot_directory, "layer_visualization")
user_prediction_directory = os.path.join(plot_directory, "user_prediction")
pytorch_models_directory = "./pytorch_checkpoints_and_models"
results_directory = "./rnn_results"
autotune_preprocessed_directory = os.path.join(base_directory, "autotune_preprocessed_pyin")

# path to a specific query csv file
metadata_csv = os.path.join(base_directory, "intonation.csv")
