
"""Given outputs of rnn.py saved to disk, synthesizes the pitch-corrected vocals track"""

import librosa
import numpy as np
import pickle as pkl

import dataset_analysis
from globals import *

model = "51"
base_dir = "/Users/scwager/Downloads/autotune_temp/results" + model + "/rnn_results"

downsample_fs = global_fs

performance_ids = [["12345", "678"], ["23456", "678"]]

for _, item in enumerate(performance_ids):
    id = item[0]
    arr_id = item[1]
    epoch = "13"
    pkl_fpath = os.path.join(base_dir, id + ".pkl")
    out_fpath = os.path.join(base_dir, "testing_epoch_" + epoch + "_outputs_" + id + ".npy")
    gt_fpath = os.path.join(base_dir, "testing_epoch_" + epoch + "_labels_" + id + ".npy")

    # load ground truth and predictions
    with open(pkl_fpath, "rb") as f:
        data_dict = pkl.load(f)
    preds = np.load(out_fpath)
    labels = np.load(gt_fpath)
    preds = np.squeeze(preds)  # convert to shape < notes, shifts >
    labels = np.squeeze(labels)

    midi = data_dict['midi']
    notes = data_dict['notes']
    stacked_cqt_v = data_dict['spect_v']
    cqt_b = data_dict['spect_b']
    stacked_cqt_combined = data_dict['spect_c']
    shifted_pyin = data_dict['shifted_pyin']
    original_boundaries = data_dict['original_boundaries']

    # load the original, in-tune pitch track
    pitch_track = np.load(os.path.join(base_dir, "pyin_" + id + ".npy"))
    frames = len(pitch_track)

    # load the original audio
    vocal_stft = dataset_analysis.get_stft(os.path.join(base_dir, id + ".wav")) * 3
    vocal_audio = dataset_analysis.get_audio(os.path.join(base_dir, id + ".wav")) * 2
    backing_audio = dataset_analysis.get_audio(os.path.join(base_dir, arr_id + ".wav"))
    test_mix = vocal_audio + backing_audio  # check that the audio is correct
    test_mix = librosa.core.resample(test_mix, orig_sr=downsample_fs, target_sr=downsample_fs)
    test_mix = test_mix[:30 * downsample_fs]
    librosa.output.write_wav("/Users/scwager/Downloads/autotune_temp/test_mix_" + id + ".wav", test_mix, sr=global_fs)

    idx = 2  # index of shift version, in range [0, 7)

    # apply shifts to the original frame indices
    frame_preds = np.zeros(frames)
    frame_labels = np.zeros(frames)

    for i in range(len(labels[:, 0])):
        frame_preds[original_boundaries[i, 0]: original_boundaries[i, 1]] += preds[i, idx]
        frame_labels[original_boundaries[i, 0]: original_boundaries[i, 1]] += labels[i, idx]

    # shift pyin pitch to get the de-tuned input to the neural net, then apply correction (negative of learned shift)
    shifted_pitch_track = np.copy(pitch_track)
    corrected_pitch_track = np.copy(pitch_track)
    for i in range(len(labels[:, 0])):
        shifted_pitch_track[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                    np.power(2, -labels[i, idx] / 12.0)
        corrected_pitch_track[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                    np.power(2, (-labels[i, idx] + preds[i, idx]) / 12.0)

    # do the same with the audio
    shifted_audio = []
    corrected_audio = []
    shifted_audio_stft = np.copy(vocal_stft)
    corrected_audio_stft = np.copy(vocal_stft)
    for i in range(len(labels[:, 0])):
        # for i, note in enumerate(notes):
        note_stft = np.array(vocal_stft[:, original_boundaries[i, 0]:original_boundaries[i, 1]])
        if note_stft.shape[0] == 0 or note_stft.shape[1] == 0:
            print("empty note")
            continue
        note_rt = librosa.istft(note_stft, hop_length=hopSize, center=False)
        shifted_note_rt = librosa.effects.pitch_shift(note_rt, sr=global_fs, n_steps=labels[i, idx])
        corrected_note_rt = librosa.effects.pitch_shift(note_rt, sr=global_fs, n_steps=labels[i, idx] - preds[i, idx])
        shifted_note_stft = librosa.stft(shifted_note_rt, n_fft=frameSize, hop_length=hopSize, center=False)
        corrected_note_stft = librosa.stft(corrected_note_rt, n_fft=frameSize, hop_length=hopSize, center=False)
        shifted_audio_stft[:, original_boundaries[i, 0]:original_boundaries[i, 1]] = shifted_note_stft
        corrected_audio_stft[:, original_boundaries[i, 0]:original_boundaries[i, 1]] = corrected_note_stft
    shifted_audio = librosa.istft(shifted_audio_stft, hop_length=hopSize, center=False)
    corrected_audio = librosa.istft(corrected_audio_stft, hop_length=hopSize, center=False)

    min_len = min([len(shifted_audio), len(backing_audio), len(corrected_audio)])
    shifted_mix = np.array(shifted_audio[:min_len]) + backing_audio[:min_len]  # check that the audio is correct
    corrected_mix = np.array(corrected_audio[:min_len]) + backing_audio[:min_len]
    shifted_mix = librosa.core.resample(shifted_mix, orig_sr=global_fs, target_sr=downsample_fs)
    corrected_mix = librosa.core.resample(corrected_mix, orig_sr=global_fs, target_sr=downsample_fs)
    shifted_mix = shifted_mix[:30 * downsample_fs]
    corrected_mix = corrected_mix[:30 * downsample_fs]
    librosa.output.write_wav(
            "/Users/scwager/Downloads/autotune_temp/shifted_mix_" + id + ".wav", shifted_mix, sr=downsample_fs)
    librosa.output.write_wav(
            "/Users/scwager/Downloads/autotune_temp/corrected_mix_" + id + ".wav", corrected_mix, sr=downsample_fs)
