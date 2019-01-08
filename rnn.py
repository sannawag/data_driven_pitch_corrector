#!/usr/bin/python3.6

"""
Main program for training a model to predict note-wise pitch corrections.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# other imports from parallel packages
import dataset_analysis
from globals import *
import utils

import argparse
import bokeh.plotting as bplt
from bokeh.models import Span
from skimage.filters import threshold_mean
import calendar
import csv
import librosa
import librosa.display
import math
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


import logging
logger = logging.getLogger(__name__)


class ConvRNN(nn.Module):

    def __init__(self, hidden_size, num_layers):

        # parameters for CRNN
        super(ConvRNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(2, 128, kernel_size=5, stride=(1, 2), padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=(1, 2), padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 8, kernel_size=(48, 3), stride=1, padding=(24, 1))
        self.conv6 = nn.Conv2d(8, 1, kernel_size=1, stride=1)

        # parameters for GRU
        self.num_layers = num_layers  # rnn layers
        self.hidden_size = hidden_size
        self.hidden = None
        self.hidden_out = None
        self.rnn = nn.GRU(input_size=289, hidden_size=64, num_layers=1)  # input size depends on CNN structure
        self.linear = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

        # initialize parameters for CNN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_hidden(self, device, batch_size):  # for GRU
        hidden = torch.zeros(self.num_layers, batch_size, 64, device=device)
        return torch.nn.init.normal_(hidden) * 0.0001, None

    def forward(self, stft_v, stft_b, stft_c, register_hooks=False, plot=False, layer_plot_directory="",
                plot_title_ext=""):

        ts = str(calendar.timegm(time.gmtime()))  # timestamp for plotting

        # plot the three input channels for two different
        if plot is True:
            utils.plot_layer(stft_v, ts + "_step_0_voice", dim=3, layer_plot_directory=layer_plot_directory)
            utils.plot_layer(stft_b, ts + "_step_0_back", dim=3, layer_plot_directory=layer_plot_directory)
            utils.plot_layer(stft_c, ts + "_step_0_difference", dim=3, layer_plot_directory=layer_plot_directory)

        # x = torch.stack((stft_v, stft_b, stft_c))  # shape <channels, seq length, batch size, Fdim>
        x = torch.stack((stft_v, stft_b))  # shape <channels, seq length, batch size, Fdim>
        x = x.permute(2, 0, 3, 1)  # <batch size, channels (2), Fdim, seq length>
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1, x.size(3))  # remove dimension
        x = x.permute(2, 0, 1)  # <seq length, batch, dim>

        x, self.hidden = self.rnn(x, self.hidden.detach())  # feedforward RNN
        x = x[-1:, :, :]  # keep only the last sample
        x = self.linear(x)  # linear layer with one-dimensional output
        # x = self.tanh(x)

        x = x.view(x.shape[0], x.shape[1])  # remove additional dimension

        return x


def save_outputs(results_dir, user_prediction_dir, epoch, perf, outputs, labels_tensor, original_boundaries,
                 training):
    """Save predictions and labels for one shift of every song in batch"""
    try:
        f_start = "training" if training is True else "testing"
        ts = str(calendar.timegm(time.gmtime()))
        bplt.output_file(os.path.join(
                user_prediction_dir, f_start + "_epoch_" + str(epoch) + "_" + ts + "_" + perf + ".html"))

        # load the original, in-tune pitch track
        pitch_track = np.load(os.path.join(pyin_directory, perf + ".npy"))
        frames = len(pitch_track)

        # convert to shape < notes, shifts >
        outputs = np.squeeze(outputs)
        labels = np.squeeze(labels_tensor)

        # plot the shifts after applying them to the original frame indices
        frame_outputs_0 = np.zeros(frames)
        frame_labels_0 = np.zeros(frames)
        frame_outputs_5 = np.zeros(frames)
        frame_labels_5 = np.zeros(frames)
        for i in range(len(labels[:, 0])):
            frame_outputs_0[original_boundaries[i, 0]: original_boundaries[i, 1]] += outputs[i, 0]
            frame_labels_0[original_boundaries[i, 0]: original_boundaries[i, 1]] += labels[i, 0]
            frame_outputs_5[original_boundaries[i, 0]: original_boundaries[i, 1]] += outputs[i, 5]
            frame_labels_5[original_boundaries[i, 0]: original_boundaries[i, 1]] += labels[i, 5]
        s1 = bplt.figure(title="Pitch shifts: ground truth versus predictions shift 0")
        s1.line(np.arange(len(frame_labels_0)), frame_labels_0, color="red")
        s1.line(np.arange(len(frame_outputs_0)), frame_outputs_0, color="blue")
        s2 = bplt.figure(title="Pitch shifts: ground truth versus predictions shift 5")
        s2.line(np.arange(len(frame_labels_5)), frame_labels_5, color="red")
        s2.line(np.arange(len(frame_outputs_5)), frame_outputs_5, color="blue")

        # shift pitch to get the de-tuned input to the neural net, then apply correction (negative of learned shift)
        shifted_pitch_track_0 = np.copy(pitch_track)
        corrected_pitch_track_0 = np.copy(pitch_track)
        shifted_pitch_track_5 = np.copy(pitch_track)
        corrected_pitch_track_5 = np.copy(pitch_track)
        for i in range(len(labels[:, 0])):
            shifted_pitch_track_0[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                        np.power(2, max_semitone * labels[i, 0] / 12.0)
            corrected_pitch_track_0[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                        np.power(2, max_semitone * (labels[i, 0] - outputs[i, 0]) / 12.0)
            shifted_pitch_track_5[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                        np.power(2, max_semitone * labels[i, 5] / 12.0)
            corrected_pitch_track_5[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                        np.power(2, max_semitone * (labels[i, 5] - outputs[i, 5]) / 12.0)
        s3 = bplt.figure(title="Original input versus de-tuned input before and after pitch correction shift 0",
                         y_range=(np.min(pitch_track[pitch_track > 10] - 50), np.max(pitch_track) + 50))
        s3.line(np.arange(frames), pitch_track, color='black')
        s3.line(np.arange(frames), shifted_pitch_track_0, color='red')
        s3.line(np.arange(frames), corrected_pitch_track_0, color='green')
        s4 = bplt.figure(title="Original input versus de-tuned input before and after pitch correction shift 5",
                         y_range=(np.min(pitch_track[pitch_track > 10] - 50), np.max(pitch_track) + 50))
        s4.line(np.arange(frames), pitch_track, color='black')
        s4.line(np.arange(frames), shifted_pitch_track_5, color='red')
        s4.line(np.arange(frames), corrected_pitch_track_5, color='green')

        bplt.save(bplt.gridplot([s1], [s2], [s3], [s4]))

        np.save(os.path.join(
            results_dir, f_start + "_epoch_" + str(epoch) + "_outputs" + "_" + perf), outputs)
        np.save(os.path.join(
            results_dir, f_start + "_epoch_" + str(epoch) + "_labels" + "_" + perf), labels)
    except Exception as e:
        logger.info("exception in save_outputs {0} skipping song {1}".format(e, perf))
        return


class AutotuneDataset(Dataset):
    """Customized PyTorch Dataset that treats one song with all its pitch-shifted versions as a data sample.
    This function is used as input to the DataLoader."""

    def __init__(self, performance_list, num_shifts, metadata_csv, plot=False, freeze=False):
        self.performance_list = performance_list
        self.plot = plot
        self.num_shifts = num_shifts
        self.freeze = freeze
        self.mode = "cqt"
        # load pre-defined shifts to fix a few examples
        self.frozen_shifts = np.load("frozen_shifts.npy")
        # load the csv
        self.arr_keys = {}  # store backing track arrangement key corresponding to performance ID
        with open(metadata_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.arr_keys[row[0].strip()] = row[1].strip()
                    line_count += 1

    def __len__(self):
        return len(self.performance_list)

    def loaditem(self, fpath):
        """
        Loads a previously processed item and scrambles the shifts across the notes
        :param idx: performance index
        :return: result for __getitem__
        """
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
        # scramble
        stacked_cqt_v = np.copy(data_dict['spect_v'])
        stacked_cqt_combined = np.copy(data_dict['spect_c'])
        shifted_pyin = np.copy(data_dict['shifted_pyin'])

        for i, note in enumerate(data_dict['notes']):
            if self.freeze is True:
                order = np.hstack(
                    (np.arange(3), np.random.choice(self.num_shifts - 3, self.num_shifts - 3, replace=False) + 3))
            else:
                order = np.random.choice(self.num_shifts, self.num_shifts, replace=False)
            data_dict['shifts_gt'][:, i] = data_dict['shifts_gt'][order, i]
            for j in range(self.num_shifts):
                stacked_cqt_v[note[0]:note[1], j, :] = data_dict['spect_v'][note[0]:note[1], order[j], :]
                stacked_cqt_combined[note[0]:note[1], j, :] = data_dict['spect_c'][note[0]:note[1], order[j], :]
                shifted_pyin[j, note[0]:note[1]] = data_dict['shifted_pyin'][order[j], note[0]:note[1]]

        data_dict['spect_v'] = stacked_cqt_v
        data_dict['spect_c'] = stacked_cqt_combined
        data_dict['shifted_pyin'] = shifted_pyin

        return data_dict

    def __getitem__(self, idx):
        """
        :param keep: number of songs to keep, less or equal to the number of pitch shifts
        :return: a list of length num_performance_shifts.
        Each item is a tuple with the labels <seq len, shifts>, input <seq len, shifts, fdim>, key (str)
        """
        fpath = os.path.join(autotune_preprocessed_directory, self.performance_list[idx] + ".pkl")
        if not os.path.exists(fpath):
            try:
                pyin = np.load(os.path.join(pyin_directory, self.performance_list[idx] + ".npy"))
                # load stft of vocals, keep complex values in order to use istft later for pitch shifting
                stft_v = dataset_analysis.get_stft(
                    os.path.join(vocals_directory, self.performance_list[idx] + ".wav")).T
                # load cqt of backing track
                cqt_b = np.abs(dataset_analysis.get_cqt(os.path.join(backing_tracks_directory,
                               self.arr_keys[self.performance_list[idx]] + ".wav"))).T
                # truncate pitch features to same length
                frames = min(cqt_b.shape[0], stft_v.shape[0], len(pyin))
                pyin = pyin[:frames]
                stft_v = stft_v[:frames, :]
                cqt_b = cqt_b[:frames, :]
                original_boundaries = np.arange(frames).astype(np.int64)  # store the original indices of the notes here
                # find locations of note onsets using pYIN
                min_note_frames = 24  # half second
                audio_beginnings = np.array([i for i in range(frames - min_note_frames)  # first nonzero frames
                                             if i == 0 and pyin[i] > 0 or i > 0 and pyin[i] > 0 and pyin[i - 1] == 0])
                if self.plot is True:
                    utils.reset_directory("./plots")
                    bplt.output_file(os.path.join("./plots", "note_parse_" + self.performance_list[idx]) + ".html")
                    s1 = bplt.figure(title="note parse")
                    s1.line(np.arange(len(pyin)), pyin)
                    for i, ab in enumerate(audio_beginnings):
                        loc = Span(location=ab, dimension='height', line_color='green')
                        s1.add_layout(loc)
                # discard silent frames
                silent_frames = np.ones(frames)
                silent_frames[np.where(pyin < 1)[0]] *= 0
                pyin = pyin[silent_frames.astype(bool)]
                stft_v = stft_v[silent_frames.astype(bool), :]
                cqt_b = cqt_b[silent_frames.astype(bool), :]
                original_boundaries = original_boundaries[silent_frames.astype(bool)]
                audio_beginnings = [n - np.sum(silent_frames[:n] == 0) for _, n in enumerate(audio_beginnings)]
                frames = len(pyin)
                audio_endings = np.hstack((audio_beginnings[1:], frames - 1))
                # merge notes that are too short
                note_beginnings = []
                note_endings = []
                min_note_frames = 24
                start_note = end_note = 0
                while start_note < len(audio_beginnings):
                    note_beginnings.append(audio_beginnings[start_note])
                    while (audio_endings[end_note] - audio_beginnings[start_note] < min_note_frames and
                           end_note < len(audio_endings) - 1):
                        end_note += 1
                    note_endings.append(audio_endings[end_note])
                    start_note = end_note + 1
                    end_note = start_note
                # check that the last note is long enough
                while note_endings[-1] - note_beginnings[-1] < min_note_frames:
                    del note_beginnings[-1]
                    del note_endings[-2]
                notes = np.array([note_beginnings, note_endings]).T
                # one minor issue
                if notes[-1, 1] > frames - 1:
                    notes[-1, 1] = frames - 1
                if self.plot is True:
                    s2 = bplt.figure(title="note parse of active frames")
                    s2.line(np.arange(len(pyin)), pyin)
                    for i, ab in enumerate(note_beginnings):
                        loc = Span(location=ab, dimension='height', line_color='green')
                        s2.add_layout(loc)
                    for i, ab in enumerate(note_endings):
                        loc = Span(location=ab+1, dimension='height', line_color='red', line_dash='dotted')
                        s2.add_layout(loc)
                    bplt.save(bplt.gridplot([[s1, s2]], toolbar_location=None))
                # store the original indices of the notes
                original_boundaries = np.array([original_boundaries[notes[:, 0]], original_boundaries[notes[:, 1]]]).T
                # compute shifts for every note in every version in the batch (num_shifts)
                note_shifts = np.random.rand(self.num_shifts, notes.shape[0]) * 2 - 1  # all shift combinations
                if self.freeze is True:
                    note_shifts[:3, :] = self.frozen_shifts[:3, :note_shifts.shape[1]]
                # compute the framewise shifts
                frame_shifts = np.zeros((self.num_shifts, frames))  # this will be truncated later
                for i in range(self.num_shifts):
                    for j in range(len(notes)):
                        # only shift the non-silent frames between the note onset and note offset
                        frame_shifts[i, notes[j][0]:notes[j][1]] = note_shifts[i][j]
                # de-tune the pYIN pitch tracks and STFT of vocals
                shifted_pyin = np.vstack([pyin] * self.num_shifts) * np.power(2, max_semitone * frame_shifts / 12)
                # de-tune the vocals stft and vocals cqt
                stacked_cqt_v = np.zeros((frames, self.num_shifts, cqt_params['total_bins']))
                for i, note in enumerate(notes):
                    note_stft = np.array(stft_v[note[0]:note[1], :]).T
                    note_rt = librosa.istft(note_stft, hop_length=hopSize, center=False)
                    for j in range(self.num_shifts):
                        shifted_note_rt = librosa.effects.pitch_shift(note_rt, sr=global_fs, n_steps=note_shifts[j, i])
                        stacked_cqt_v[note[0]:note[1], j, :] = np.abs(librosa.core.cqt(
                                shifted_note_rt, sr=global_fs, hop_length=hopSize, n_bins=cqt_params['total_bins'],
                                bins_per_octave=cqt_params['bins_per_8va'], fmin=cqt_params['fmin']))[:, 4:-4].T
                # get the data into the proper format and shape for tensors
                cqt_b_binary = np.copy(cqt_b)  # copy single-channel CQT for binarization
                # need to repeat the backing track for the batch
                cqt_b = np.stack([cqt_b] * self.num_shifts, axis=1)
                # third channel
                stacked_cqt_v_binary = np.copy(stacked_cqt_v)
                for i in range(self.num_shifts):
                    thresh = threshold_mean(stacked_cqt_v_binary[:, i, :])
                    stacked_cqt_v_binary[:, i, :] = (stacked_cqt_v_binary[:, i, :] > thresh).astype(np.float)
                thresh = threshold_mean(cqt_b_binary)
                cqt_b_binary = (cqt_b_binary > thresh).astype(np.float)
                stacked_cqt_b_binary = np.stack([cqt_b_binary] * self.num_shifts, axis=1)
                stacked_cqt_combined = np.abs(stacked_cqt_v_binary - stacked_cqt_b_binary)

                data_dict = dict()
                data_dict['notes'] = notes
                data_dict['spect_v'] = stacked_cqt_v
                data_dict['spect_b'] = cqt_b
                data_dict['spect_c'] = stacked_cqt_combined
                data_dict['shifted_pyin'] = shifted_pyin
                data_dict['shifts_gt'] = note_shifts
                data_dict['original_boundaries'] = original_boundaries
                data_dict['perf_id'] = self.performance_list[idx]

                with open(fpath, "wb") as f:
                    pickle.dump(data_dict, f)  # save for future epochs
            except Exception as e:
                logger.info("exception in dataset {0} skipping song {1}".format(e, self.performance_list[idx]))
                return None
        else:
            # pre-processing has already been computed: load from file
            try:
                data_dict = self.loaditem(fpath)
            except Exception as e:
                logger.info("exception in dataset {0} skipping song {1}".format(e, self.performance_list[idx]))
                return None
        try:
            # now format the numpy arrays into torch tensors with note-wise splits
            data_dict['spect_v'] = torch.Tensor(data_dict['spect_v'])
            data_dict['spect_b'] = torch.Tensor(data_dict['spect_b'])
            data_dict['spect_c'] = torch.Tensor(data_dict['spect_c'])
            data_dict['shifted_pyin'] = torch.Tensor(data_dict['shifted_pyin'].T)
            data_dict['shifts_gt'] = torch.Tensor(data_dict['shifts_gt'].T)
            # adjust dimension of note shifts
            data_dict['shifts_gt'].unsqueeze_(1)

            # split full songs into sequences
            split_sizes = tuple(np.append(
                    np.diff(data_dict['notes'][:, 0]), data_dict['notes'][-1, 1] - data_dict['notes'][-1, 0] + 1))
            data_dict['spect_v'] = torch.split(data_dict['spect_v'], split_size_or_sections=split_sizes, dim=0)
            data_dict['spect_b'] = torch.split(data_dict['spect_b'], split_size_or_sections=split_sizes, dim=0)
            data_dict['spect_c'] = torch.split(data_dict['spect_c'], split_size_or_sections=split_sizes, dim=0)
            data_dict['shifted_pyin'] = torch.split(data_dict['shifted_pyin'], split_size_or_sections=split_sizes,
                                                    dim=0)
        except Exception as e:
            logger.info("exception in dataset {0} skipping song {1}".format(e, self.performance_list[idx]))
            return None
        return data_dict


def get_dataset(data_list, songs_per_batch, num_shifts, device, mode="testing", workers_on_gpu=2, freeze=False):
    dataset = AutotuneDataset(data_list, num_shifts=num_shifts, metadata_csv=metadata_csv, plot=False, freeze=freeze)
    return dataset


def save_checkpoint(state, is_best, latest_filename, best_filename):
    """PyTorch helper function: saves the model to file and also overwrites model_best for early stopping"""

    torch.save(state, latest_filename)
    logger.info("=> saved latest checkpoint at {0}".format(latest_filename))
    if is_best is True:
        torch.save(state, best_filename)
        logger.info("=> saved best checkpoint at {0}".format(best_filename))


def restore_checkpoint(resume, resume_file, device, model, optimizer):
    # initialize and, optionally, restore a previous checkpoint
    best_prec1 = 1e100  # smallest loss so far
    start_epoch = 1
    training_losses = []
    validation_losses = []
    if resume is True:
        if os.path.isfile(resume_file):
            logger.info("=> loading checkpoint '{}'".format(resume_file))
            if str(device) == 'cpu':
                checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            training_losses = checkpoint['training_losses']
            validation_losses = checkpoint['validation_losses']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{0}' (epoch {1})".format(resume_file, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{0}'".format(resume_file))
    return best_prec1, start_epoch, training_losses, validation_losses, model, optimizer


def split_into_training_validation_test(performance_list, boundary_arr_id="12345"):
    arr_keys = {}  # store backing track arrangement key corresponding to performance ID
    # user_ids = {}  # store the user ids corresponding to performance ID
    with open(metadata_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                arr_keys[row[0].strip()] = row[1].strip()
                # user_ids[row[0].strip()] = row[2].strip()
                line_count += 1
    validation_list = [p for i, p in enumerate(performance_list) if arr_keys[p] <= boundary_arr_id]
    test_list = validation_list  # TODO: create a separate test list
    training_list = sorted(list(set(performance_list) - set(validation_list)))
    return training_list, validation_list, test_list


class Program:
    def __init__(self, extension, hidden_size, global_dropout, learning_rate, resume, small_dataset, max_norm,
                 num_layers, num_shifts, epochs, report_step, sandbox):
        self.extension = extension
        self.hidden_size = hidden_size
        self.global_dropout = global_dropout
        self.learning_rate = learning_rate
        self.resume = resume
        self.small_dataset = small_dataset
        self.max_norm = max_norm
        self.num_layers = num_layers
        self.num_shifts = num_shifts
        self.epochs = epochs
        self.report_step = report_step
        self.sandbox = sandbox
        self.is_best = False

        # set paths, make sure all directories exist and delete results from previous runs
        self.results_root = "./results_root" + self.extension  # root directory for plots and results
        self.results_directory = os.path.join(self.results_root, "rnn_results")
        self.plot_directory = os.path.join(self.results_root, "plots")
        self.parameter_plot_directory = os.path.join(self.plot_directory, "parameter_visualization")
        self.layer_plot_directory = os.path.join(self.plot_directory, "layer_visualization")
        self.user_prediction_directory = os.path.join(self.plot_directory, "user_prediction")
        self.test_prediction_directory = os.path.join(self.user_prediction_directory, "test_prediction")
        self.test_results_directory = os.path.join(self.results_directory, "test")
        utils.reset_directory(self.results_root, empty=False)
        utils.reset_directory(self.results_directory, empty=True)
        utils.reset_directory(self.user_prediction_directory, empty=True)
        utils.reset_directory(self.layer_plot_directory, empty=True)
        utils.reset_directory(self.test_prediction_directory, empty=True)
        utils.reset_directory(self.test_results_directory, empty=True)
        utils.reset_directory(pytorch_models_directory, empty=False)
        utils.reset_directory(autotune_preprocessed_directory, empty=False)
        logger.info("preprocessed dir: {0}".format(autotune_preprocessed_directory))
        # pytorch checkpoint directories
        self.resume_file = os.path.join(pytorch_models_directory, 'model_best' + self.extension + '.pth.tar')
        # save latest model parameters to this checkpoint
        self.latest_checkpoint_file = os.path.join(
            pytorch_models_directory, 'checkpoint_rnn' + self.extension + '.pth.tar')
        # save model parameters with best validation loss to this checkpoint
        self.best_checkpoint_file = os.path.join(pytorch_models_directory, 'model_best' + self.extension + '.pth.tar')

        # gpu versus cpu device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model
        self.model = ConvRNN(hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        utils.print_param_sizes(self.model)

        # error and loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # initialize training parameters or load them from checkpoint if the file exists and resume is set to True
        self.best_prec1, self.start_epoch, self.training_losses, self.validation_losses, self.model, self.optimizer = \
            restore_checkpoint(self.resume, self.resume_file, self.device, self.model, self.optimizer)

        # load the performance indices for the datasets: all ids that have an instance in all directories
        performance_list = sorted(list(
            set([f[:-4] for f in os.listdir(pyin_directory) if "npy" in f]) &
            set([f[:-4] for f in os.listdir(back_chroma_directory) if "npy" in f])))

        # leave boundary_arr_id to default if using full dataset
        training_list, validation_list, test_list = split_into_training_validation_test(performance_list,
                boundary_arr_id="3769415_3769415")

        # keep only a subset for development purposes
        if self.small_dataset is True:
            validation_list = validation_list[:1] + training_list[:1]
            training_list = training_list[:1]
            test_list = training_list[:1]
            logger.info("training list {0} validation list {1}".format(training_list, validation_list))
        else:
            logger.info("training list {0} validation list {1}".format(len(training_list), len(validation_list)))
        # build custom datasets
        freeze = True if self.small_dataset is True else False  # freeze some note-wise shifts to check program accuracy
        logger.info("fixing some of the shifts across all songs: {0}".format(freeze))
        self.training_dataset = get_dataset(data_list=training_list, num_shifts=self.num_shifts,
                songs_per_batch=1, mode="training", device=self.device, freeze=freeze)
        self.validation_dataset = get_dataset(data_list=validation_list, num_shifts=self.num_shifts,
                songs_per_batch=1, mode="testing", device=self.device, freeze=freeze)
        self.test_dataset = get_dataset(data_list=test_list, num_shifts=self.num_shifts,
                songs_per_batch=1, mode="testing", device=self.device, freeze=freeze)

    def eval(self, data_dict, save_song_outputs=False, plot=False):
        """ Evaluation function for one batch"""
        # set up and initialize
        # self.model.eval()  #
        with torch.no_grad():
            batch_size = data_dict['shifts_gt'].size(2)  # the batch size can vary at the last sample of an epoch
            # reset hidden states
            self.model.hidden, self.model.hidden_out = self.model.init_hidden(device=self.device, batch_size=batch_size)
            outputs_list = []  # Store some example outputs here
            batch_loss = 0.0
            num_notes = len(data_dict['shifted_pyin'])
            for j in range(num_notes):
                # feedforward
                outputs = self.model(data_dict['spect_v'][j].to(self.device), data_dict['spect_b'][j].to(self.device),
                        data_dict['spect_c'][j].to(self.device), register_hooks=False,
                        plot=True if plot and j == 3 else False, layer_plot_directory=self.layer_plot_directory)
                # loss
                loss = self.criterion(outputs, data_dict['shifts_gt'][j].to(self.device))
                batch_loss += loss.item()
                if save_song_outputs:
                    outputs_list.append(outputs.detach().cpu().numpy())
        # return average loss over the batch
        return np.array(outputs_list), batch_loss / num_notes

    def train(self, data_dict, save_song_outputs=False, plot=False, register_hooks=False):
        """ Training function for one batch"""
        # set up and initialize
        self.model.train()  # training mode (use dropout)
        batch_size = data_dict['shifts_gt'].size(2)  # the batch size can vary at the last sample of an epoch
        # reset hidden states
        self.model.hidden, self.model.hidden_out = self.model.init_hidden(device=self.device, batch_size=batch_size)
        outputs_list = []  # Store some example outputs here
        batch_loss = 0.0
        num_notes = len(data_dict['shifted_pyin'])
        for j in range(num_notes):
            # feedforward
            outputs = self.model(data_dict['spect_v'][j].to(self.device), data_dict['spect_b'][j].to(self.device),
                    data_dict['spect_c'][j].to(self.device), register_hooks=register_hooks,
                    plot=True if plot and j < 2 else False, layer_plot_directory=self.layer_plot_directory,
                    plot_title_ext="training_" + str(j))
            # loss
            loss = self.criterion(outputs, data_dict['shifts_gt'][j].to(self.device))
            loss.backward(retain_graph=True)  # compute the gradients
            batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            if save_song_outputs:
                outputs_list.append(outputs.detach().cpu().numpy())
        # return average loss over the batch
        self.optimizer.step()
        self.model.zero_grad()  # reset gradients
        return np.array(outputs_list), batch_loss / num_notes

    def test_iters(self, epoch, dataloader):
        total_validation_loss = 0.0
        validation_batch_count = 0
        for j, data_dict in enumerate(dataloader):
            if data_dict is None:
                continue
            save_song = True if j < 20 else False
            plot = False
            outputs, loss = self.eval(data_dict, save_song_outputs=save_song, plot=plot)
            validation_batch_count += 1
            total_validation_loss += loss
            # save sample outputs
            if save_song:
                save_outputs(self.test_results_directory, self.test_prediction_directory, epoch,
                             data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                             data_dict['original_boundaries'], training=False)
            del data_dict; del outputs  # clear memory
        return total_validation_loss / validation_batch_count

    def validate_iters(self, epoch, dataloader):
        total_validation_loss = 0.0
        validation_batch_count = 0
        for j, data_dict in enumerate(dataloader):
            if data_dict is None:
                continue
            save_song = True if j % self.report_step < 7 else False
            plot = False
            outputs, loss = self.eval(data_dict, save_song_outputs=save_song, plot=plot)
            validation_batch_count += 1
            total_validation_loss += loss
            # save sample outputs
            if save_song:
                save_outputs(self.results_directory, self.user_prediction_directory, epoch,
                             data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                             data_dict['original_boundaries'], training=False)
            del data_dict; del outputs  # clear memory
        return total_validation_loss / validation_batch_count

    def train_iters(self):
        logger.info("number of parameters {0}".format(utils.get_n_params(self.model)))
        for epoch in range(self.start_epoch, self.epochs + 1):

            start_time = time.time()
            total_training_loss = 0.0
            training_batch_count = 0
            counter = 0
            # training
            for i, data_dict in enumerate(self.training_dataset):
                if i < 2 or i % 30 == 0 or counter < 1 or counter % 30 == 0:
                    logger.info("epoch {0} step {1} counter {2}".format(epoch, i, counter))
                if data_dict is None:
                    continue
                save_song = True if i % self.report_step < 5 else False  # return the predictions from train() if true
                # train on one batch, interating through the segments
                outputs, loss = self.train(data_dict, save_song_outputs=save_song,
                                           plot=True if i == 0 or i == 1 else False, register_hooks=False)
                training_batch_count += 1
                total_training_loss += loss
                # save sample outputs
                if save_song:
                    save_outputs(self.results_directory, self.user_prediction_directory, epoch,
                                 data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                                 data_dict['original_boundaries'], training=True)
                del data_dict; del outputs  # clear memory

                # validate and report losses every report_step and at end of epoch. Save the checkpoint if it is better
                if counter % self.report_step == 0:
                    # iterate through the full validation set
                    mean_validation_loss = self.validate_iters(epoch, self.validation_dataset)
                    mean_training_loss = total_training_loss / training_batch_count  # running average of training loss
                    # report current losses and print list of losses so far
                    logger.info("***********************************************************************************")
                    logger.info("{0}: Training and validation loss epoch {1} step {2} : {3} {4}".format(
                        self.extension, epoch, i, mean_training_loss, mean_validation_loss))

                    self.training_losses.append(mean_training_loss)
                    self.validation_losses.append(mean_validation_loss)
                    logger.info("***********************************************************************************")
                    logger.info("Training and validation losses so far:\n{0}\n{1}".format(
                        self.training_losses, self.validation_losses))

                    # plot the loss curves
                    bplt.output_file(os.path.join(self.plot_directory, "rnn_losses.html"))
                    fig_tr = bplt.figure(title="Training losses")
                    fig_ev = bplt.figure(title="Evaluation losses")
                    fig_cb = bplt.figure(title="Training and evaluation losses")
                    fig_fx = bplt.figure(title="Losses with fixed y-axis range", y_range=[0, 6.0e-4])
                    fig_tr.circle(np.arange(len(self.training_losses)), self.training_losses, color="red")
                    fig_ev.circle(np.arange(len(self.validation_losses)), self.validation_losses, color="red")
                    fig_cb.circle(np.arange(len(self.training_losses)), self.training_losses, color="green")
                    fig_cb.circle(np.arange(len(self.validation_losses)), self.validation_losses, color="orange")
                    fig_fx.circle(np.arange(len(self.training_losses)), self.training_losses, color="green")
                    fig_fx.circle(np.arange(len(self.validation_losses)), self.validation_losses, color="orange")
                    bplt.save(bplt.gridplot([fig_tr, fig_ev], [fig_cb, fig_fx]))

                    # save model and replace best if necessary
                    logger.info("is_best before {0} mean loss {1} best prec {2}".format(self.is_best, mean_validation_loss, self.best_prec1))

                    self.is_best = True if mean_validation_loss < self.best_prec1 else False
                    self.best_prec1 = min(mean_validation_loss, self.best_prec1)
                    logger.info("is_best after {0} mean loss {1} best prec {2}".format(self.is_best, mean_validation_loss, self.best_prec1))

                    if self.sandbox is False:
                        save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                         'best_prec1': self.best_prec1, 'optimizer': self.optimizer.state_dict(),
                                         'training_losses': self.training_losses,
                                         'validation_losses': self.validation_losses}, self.is_best,
                                        latest_filename=self.latest_checkpoint_file,
                                        best_filename=self.best_checkpoint_file)
                counter += 1
            # simulated annealing
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] *= 0.998
            logger.info("--- {0} time elapsed for one epoch ---".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN for pitch correction predictions')

    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--report_step', default=500, type=int, help='Every $ training songs, validate, save, report')
    parser.add_argument('--report_minibatch_idx', default=5, type=int, help='Report within batch')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--num_shifts', default=7, type=int, help='number of shifts per song')
    parser.add_argument('--training_songs_per_batch', default=1, type=int, help='batch size for training. Only use 1')
    parser.add_argument('--test_songs_per_batch', default=1, type=int, help='batch size for testing. Only use 1')
    parser.add_argument('--global_dropout', default=0, type=float, help="p(dropped), constant across layers")
    parser.add_argument('--max_norm', default=100.0, type=float, help="gradient clipping parameter")
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size for first RNN layer')
    parser.add_argument('--linear_size_1', default=256, type=int, help='size of layer between input and RNN')
    parser.add_argument('--linear_size_2', default=256, type=int, help='size of layer between RNN and output')
    parser.add_argument('--num_layers', default=1, type=int, help='number of RNN layers')
    parser.add_argument('--resume', default=True, type=utils.str2bool, help='resume from previous checkpoint if true')
    parser.add_argument('--sandbox', default=False, type=utils.str2bool, help='experimental mode: do not save model')
    parser.add_argument('--small_dataset', default=False, type=utils.str2bool, help='use a few songs to test program')
    parser.add_argument('--run_training', default=True, type=utils.str2bool, help='run training and validaton')
    parser.add_argument('--run_testing', default=False, type=utils.str2bool, help='run testing')
    parser.add_argument('--extension', default="", help='extension to various files for this experiment')

    args = parser.parse_args()

    logging.basicConfig(filename='log{0}.txt'.format(args.extension), filemode='w',
                        format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.StreamHandler().setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # create and clear directories
    program = Program(extension=args.extension, hidden_size=args.hidden_size, global_dropout=args.global_dropout,
                      learning_rate=args.learning_rate, resume=args.resume, small_dataset=args.small_dataset,
                      max_norm=args.max_norm, num_layers=args.num_layers, num_shifts=args.num_shifts,
                      epochs=args.epochs, report_step=args.report_step, sandbox=args.sandbox)
    # run the program
    if args.run_training is True:
        program.train_iters()
    if args.run_testing is True:
        mean_test_loss = program.test_iters(epoch=0, dataloader=program.validation_dataset)
        logger.info("Test loss: {0}".format(mean_test_loss))
