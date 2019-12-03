#!/usr/bin/python3.6

"""
Main program for training a model to predict note-wise pitch corrections.

python rnn.py
"""

# other imports from parallel packages
import dataset_analysis
from globals import *
import utils

import argparse
import bokeh.plotting as bplt
from skimage.filters import threshold_mean
import calendar
import csv
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import traceback
import warnings
warnings.filterwarnings("ignore")


import logging
logger = logging.getLogger(__name__)


class ConvRNN(nn.Module):

    def __init__(self, hidden_size, num_layers, num_input_channels):

        # parameters for CRNN
        super(ConvRNN, self).__init__()
        self.num_input_channels = num_input_channels
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(self.num_input_channels, 128, kernel_size=5, stride=(1, 2), padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=(1, 2), padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 8, kernel_size=(192, 3), stride=1, padding=(96, 1))
        self.conv6 = nn.Conv2d(8, 1, kernel_size=1, stride=1)

        # parameters for GRU
        self.num_layers = num_layers  # rnn layers
        self.hidden_size = hidden_size
        self.hidden = None
        self.hidden_out = None
        self.rnn = nn.GRU(input_size=513, hidden_size=64, num_layers=1)  # input size depends on CNN structure
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
        if self.num_input_channels == 2:
            x = torch.stack((stft_v, stft_b))  # shape <channels, seq length, batch size, Fdim>
        else:
            x = torch.stack((stft_v, stft_b, stft_c))  # shape <channels, seq length, batch size, Fdim>
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

        x = x.view(x.shape[0], x.shape[1])  # remove additional dimension

        return x


class AutotuneDataset(Dataset):
    """Customized PyTorch Dataset that treats one song with all its pitch-shifted versions as a data sample.
    This function is used as input to the DataLoader."""

    def __init__(self, performance_list, num_shifts, metadata_csv, plot=False, realworld=False):
        self.realworld = realworld  # run program on real-world data instead of de-tuning
        self.performance_list = performance_list
        self.plot = plot
        self.num_shifts = num_shifts
        # load the csv
        self.arr_keys = {}  # stores backing track arrangement key corresponding to performance ID
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

    def getitem_realworld_cqt(self, idx):
        """
        substitute function for __getitem__ when using real-world data instead of shifting manually
        """
        pyin = np.load(os.path.join(realworld_pyin_directory, self.performance_list[idx] + ".npy"))
        # load stft of vocals. They are already truncated to one minute, so use restrict=False
        cqt_v = np.abs(dataset_analysis.get_cqt(os.path.join(realworld_vocals_directory,
                       self.performance_list[idx] + ".wav"), restrict=False)).T
        # load cqt of backing track
        cqt_b = np.abs(dataset_analysis.get_cqt(os.path.join(realworld_backing_tracks_directory,
                self.arr_keys[self.performance_list[idx]] + ".wav"), restrict=False)).T
        if self.arr_keys[self.performance_list[idx]] == "silent_backing_track":
            logger.info("loading example backing track CQT")
            cqt_b = np.load(os.path.join(realworld_backing_tracks_directory, "survive_4_back_cqt.npy"))
        # np.save(os.path.join(realworld_backing_tracks_directory, "survive_4_back_cqt.npy"), cqt_b)
        # truncate pitch features to same length
        frames = min(cqt_b.shape[0], cqt_v.shape[0], len(pyin))
        pyin = pyin[:frames]
        cqt_v = cqt_v[:frames, :]
        cqt_b = cqt_b[:frames, :]
        original_boundaries = np.arange(frames).astype(np.int64)  # store the original indices of the notes here
        # find locations of note onsets using pYIN
        pyin_notes = utils.parse_note_csv(os.path.join(realworld_pyin_directory,
                                                       self.performance_list[idx] + ".csv"))
        audio_beginnings = np.array([note[0] for note in pyin_notes])
        # discard silent frames
        silent_frames = np.zeros(frames)
        for note in pyin_notes:
            silent_frames[note[0]: note[1]] = 1
        pyin = pyin[silent_frames.astype(bool)]
        cqt_v = cqt_v[silent_frames.astype(bool), :]
        cqt_b = cqt_b[silent_frames.astype(bool), :]
        original_boundaries = original_boundaries[silent_frames.astype(bool)]
        audio_beginnings = [n - np.sum(silent_frames[:n] == 0) for _, n in enumerate(audio_beginnings)]
        frames = len(pyin)
        audio_endings = np.hstack((audio_beginnings[1:], frames - 1))
        # merge notes that are too short
        note_beginnings = audio_beginnings
        note_endings = audio_endings
        notes = np.array([note_beginnings, note_endings]).T
        # store the original indices of the notes
        original_boundaries = np.array([original_boundaries[notes[:, 0]], original_boundaries[notes[:, 1]]]).T
        # compute shifts for every note in every version in the batch (num_shifts)
        note_shifts = np.random.rand(self.num_shifts, notes.shape[0]) * 0.0002 - 0.0001  # add noise
        note_shifts_quantized = np.round(note_shifts * cqt_params['bins_per_note']).astype(int)  # CQT bins
        # compute the framewise shifts
        frame_shifts = np.zeros((self.num_shifts, frames))  # this will be truncated later
        for i in range(self.num_shifts):
            for j in range(len(notes)):
                # only shift the non-silent frames between the note onset and note offset
                frame_shifts[i, notes[j][0]:notes[j][1]] = note_shifts[i][j]
        # de-tune the pYIN pitch tracks and STFT of vocals
        shifted_pyin = np.vstack([pyin] * self.num_shifts) * np.power(2, max_semitone * frame_shifts / 12)
        # de-tune the vocals stft and vocals cqt
        stacked_cqt_v = np.zeros((frames, self.num_shifts, cqt_params['total_bins'] - 2 * cqt_params['bins_per_note']))
        for i, note in enumerate(notes):
            for j in range(self.num_shifts):
                shifted_note = np.roll(cqt_v[note[0]:note[1], :], note_shifts_quantized[j, i], axis=1)
                stacked_cqt_v[note[0]:note[1], j, :] = \
                    shifted_note[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']]
        # get the data into the proper format and shape for tensors
        cqt_b_binary = np.copy(cqt_b[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']])
        # need to repeat the backing track for the batch
        cqt_b = np.stack([cqt_b[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']]] * self.num_shifts,
                         axis=1)
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
        data_dict['pyin'] = pyin
        data_dict['shifted_pyin'] = shifted_pyin
        data_dict['shifts_gt'] = note_shifts
        data_dict['original_boundaries'] = original_boundaries
        data_dict['perf_id'] = self.performance_list[idx]
        data_dict['arr_id'] = self.arr_keys[self.performance_list[idx]]

        return data_dict

    def getitem_synthesize_cqt(self, idx):
        """Pitch shift is done simply by shifting the CQT"""
        pyin = np.load(os.path.join(pyin_directory, self.performance_list[idx] + ".npy"))
        # load stft of vocals, keep complex values in order to use istft later for pitch shifting
        cqt_v = np.abs(dataset_analysis.get_cqt(os.path.join(vocals_directory, self.performance_list[idx] + ".wav"))).T
        # load cqt of backing track
        cqt_b = np.abs(dataset_analysis.get_cqt(os.path.join(backing_tracks_directory,
                       self.arr_keys[self.performance_list[idx]] + ".wav"), use_librosa=True)).T
        # truncate pitch features to same length
        frames = min(cqt_b.shape[0], cqt_v.shape[0], len(pyin))
        pyin = pyin[:frames]
        cqt_v = cqt_v[:frames, :]
        cqt_b = cqt_b[:frames, :]
        original_boundaries = np.arange(frames).astype(np.int64)  # store the original indices of the notes here
        # find locations of note onsets using pYIN
        min_note_frames = 24  # half second
        audio_beginnings = np.array([i for i in range(frames - min_note_frames)  # first nonzero frames
                                     if i == 0 and pyin[i] > 0 or i > 0 and pyin[i] > 0 and pyin[i - 1] == 0])
        # discard silent frames
        silent_frames = np.ones(frames)
        silent_frames[np.where(pyin < 1)[0]] *= 0
        pyin = pyin[silent_frames.astype(bool)]
        cqt_v = cqt_v[silent_frames.astype(bool), :]
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
        # store the original indices of the notes
        original_boundaries = np.array([original_boundaries[notes[:, 0]], original_boundaries[notes[:, 1]]]).T
        # compute shifts for every note in every version in the batch (num_shifts)
        note_shifts = np.random.rand(self.num_shifts, notes.shape[0]) * 2 - 1
        note_shifts_quantized = np.round(note_shifts * cqt_params['bins_per_note']).astype(int)  # CQT bins
        # compute the framewise shifts
        frame_shifts = np.zeros((self.num_shifts, frames))  # this will be truncated later
        for i in range(self.num_shifts):
            for j in range(len(notes)):
                # only shift the non-silent frames between the note onset and note offset
                frame_shifts[i, notes[j][0]:notes[j][1]] = note_shifts[i][j]
        # de-tune the pYIN pitch tracks and STFT of vocals
        shifted_pyin = np.vstack([pyin] * self.num_shifts) * np.power(2, max_semitone * frame_shifts / 12)
        # de-tune the vocals stft and vocals cqt
        stacked_cqt_v = np.zeros((frames, self.num_shifts, cqt_params['total_bins'] - 2 * cqt_params['bins_per_note']))
        for i, note in enumerate(notes):
            for j in range(self.num_shifts):
                shifted_note = np.roll(cqt_v[note[0]:note[1], :], note_shifts_quantized[j, i], axis=1)
                stacked_cqt_v[note[0]:note[1], j, :] = \
                    shifted_note[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']]
        # get the data into the proper format and shape for tensors
        cqt_b_binary = np.copy(cqt_b[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']])
        # need to repeat the backing track for the batch
        cqt_b = np.stack([cqt_b[:, cqt_params['bins_per_note']: -cqt_params['bins_per_note']]] * self.num_shifts,
                         axis=1)
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
        data_dict['pyin'] = pyin
        data_dict['shifted_pyin'] = shifted_pyin
        data_dict['shifts_gt'] = note_shifts
        data_dict['original_boundaries'] = original_boundaries
        data_dict['perf_id'] = self.performance_list[idx]
        data_dict['arr_id'] = self.arr_keys[self.performance_list[idx]]

        return data_dict

    def __getitem__(self, idx):
        """
        :param keep: number of songs to keep, less or equal to the number of pitch shifts
        :return: a list of length num_performance_shifts.
        Each item is a tuple with the labels <seq len, shifts>, input <seq len, shifts, fdim>, key (str)
        """
        try:
            if self.realworld:  # predict pitch shifts for real singing
                data_dict = self.getitem_realworld_cqt(idx)
            else:  # synthesize pitch shifts
                data_dict = self.getitem_synthesize_cqt(idx)

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
            tb = traceback.format_exc()
            logger.info("exception in dataset {0} {1} skipping song {2}".format(e, tb, self.performance_list[idx]))
            return None
        return data_dict


def get_dataset(data_list, num_shifts, csv_file=metadata_csv, realworld=False):
    dataset = AutotuneDataset(data_list, num_shifts=num_shifts, metadata_csv=csv_file, plot=True, realworld=realworld)
    return dataset


def save_checkpoint(state, is_best, latest_filename, best_filename):
    """PyTorch helper function: saves the model to file and also overwrites model_best for early stopping"""

    torch.save(state, latest_filename)
    logger.info("=> saved latest checkpoint at {0}".format(latest_filename))
    if is_best is True:
        torch.save(state, best_filename)
        logger.info("=> saved best checkpoint at {0}".format(best_filename))


def restore_checkpoint(resume, resume_file, device, model, optimizer):
    """
    Initialize and, optionally, restore a previous checkpoint
    """
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
                checkpoint = torch.load(resume_file, map_location=device)
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


def split_into_training_validation_test(performance_list, boundary_id_val="12345", boundary_id_test="23456"):
    """
    Loads training data and puts aside samples for validation and test
    """
    arr_keys = {}  # store backing track arrangement key corresponding to performance ID
    with open(metadata_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                arr_keys[row[0].strip()] = row[1].strip()
                line_count += 1
    training_list = [p for i, p in enumerate(performance_list) if arr_keys[p] <= boundary_id_val]
    validation_list = [p for i, p in enumerate(performance_list) if boundary_id_val < arr_keys[p] <= boundary_id_test]
    test_list = [p for i, p in enumerate(performance_list) if arr_keys[p] >= boundary_id_test]
    return training_list, validation_list, test_list


def get_realworld_dataset(csv_fpath):
    """
    Loads real-world singing performances in order to predict pitch corrections
    """
    realworld_list = []
    with open(csv_fpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                perf_id, arr_key = row[0].strip(), row[1].strip()
                if os.path.exists(os.path.join(realworld_backing_tracks_directory, arr_key + ".wav")) and \
                        os.path.exists(os.path.join(realworld_pyin_directory, perf_id + ".npy")):
                    realworld_list.append(perf_id)
                line_count += 1
    return sorted(realworld_list)


class Program:
    def __init__(self, extension, hidden_size, global_dropout, gpu_id, learning_rate, resume, small_dataset, max_norm,
                 num_layers, num_shifts, epochs, report_step, sandbox, use_combination_channel):
        self.extension = extension
        self.hidden_size = hidden_size
        self.global_dropout = global_dropout
        self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        self.resume = resume
        self.small_dataset = small_dataset
        self.max_norm = max_norm
        self.num_layers = num_layers
        self.num_shifts = num_shifts
        self.epochs = epochs
        self.report_step = report_step
        self.sandbox = sandbox
        self.num_input_channels = 3 if use_combination_channel is True else 2
        self.is_best = False

        # set paths, make sure all directories exist and delete results from previous runs
        self.results_root = "./results_root_" + self.extension  # root directory for plots and results
        self.training_results_directory = os.path.join(self.results_root, "results_training_validation")
        self.test_results_directory = os.path.join(self.results_root, "results_test")
        self.realworld_results_directory = os.path.join(self.results_root, "results_realworld")
        self.training_plot_directory = os.path.join(self.results_root, "plots_training_validation")
        self.test_plot_directory = os.path.join(self.results_root, "plots_test")
        self.realworld_plot_directory = os.path.join(self.results_root, "plots_subjective_test")
        self.model_plot_directory = os.path.join(self.results_root, "other_plots")
        self.parameter_plot_directory = os.path.join(self.model_plot_directory, "parameter_visualization")
        self.layer_plot_directory = os.path.join(self.model_plot_directory, "layer_visualization")
        self.test_audio_output_directory = os.path.join(self.results_root, "test_audio_output")
        self.realworld_audio_output_directory = os.path.join(self.results_root, "realworld_audio_output")
        utils.reset_directory(self.test_audio_output_directory, empty=False)
        utils.reset_directory(self.realworld_audio_output_directory, empty=False)
        utils.reset_directory(self.results_root, empty=False)
        utils.reset_directory(self.training_results_directory, empty=False)
        utils.reset_directory(self.test_results_directory, empty=False)
        utils.reset_directory(self.realworld_results_directory, empty=False)
        utils.reset_directory(self.training_plot_directory, empty=False)
        utils.reset_directory(self.test_plot_directory, empty=False)
        utils.reset_directory(self.realworld_plot_directory, empty=False)
        utils.reset_directory(self.layer_plot_directory, empty=False)
        utils.reset_directory(self.parameter_plot_directory, empty=False)
        utils.reset_directory(pytorch_models_directory, empty=False)  # save pytorch checkpoints
        # pytorch checkpoint directories
        self.resume_file = os.path.join(pytorch_models_directory, 'model_best' + self.extension + '.pth.tar')
        # save latest model parameters to this checkpoint
        self.latest_checkpoint_file = os.path.join(
            pytorch_models_directory, 'checkpoint_rnn' + self.extension + '.pth.tar')
        # save model parameters with best validation loss to this checkpoint
        self.best_checkpoint_file = os.path.join(pytorch_models_directory, 'model_best' + self.extension + '.pth.tar')

        # gpu versus cpu device
        # self.device = torch.device(self.gpu_id if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: {}".format(self.device))

        # model
        self.model = ConvRNN(hidden_size=self.hidden_size, num_layers=self.num_layers,
                             num_input_channels=self.num_input_channels).to(self.device)
        utils.print_param_sizes(self.model)

        # error and loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        with open(os.path.join(self.results_root, 'program_args' + args.extension + '.txt'), 'w') as f:
            f.write(','.join(
                ["dropout " + str(self.global_dropout),
                 "hidden size " + str(self.hidden_size),
                 "learning rate " + str(self.learning_rate),
                 "max norm " + str(self.max_norm),
                 "num layers " + str(self.num_layers),
                 "num shifts " + str(self.num_shifts),
                 "\ncriterion" + str(self.criterion),
                 "optimizer " + str(self.optimizer),
                 "model " + str(self.model)]
            ))

        # initialize training parameters or load them from checkpoint if the file exists and resume is set to True
        self.best_prec1, self.start_epoch, self.training_losses, self.validation_losses, self.model, self.optimizer = \
            restore_checkpoint(self.resume, self.resume_file, self.device, self.model, self.optimizer)

        logger.info("losses: {0}, {1}".format(self.training_losses, self.validation_losses))

        # load the performance indices for the datasets
        performance_list = sorted(list(set([f[:-4] for f in os.listdir(pyin_directory) if "npy" in f])))

        # leave boundary_arr_id to default if using full dataset
        training_list, validation_list, test_list = split_into_training_validation_test(performance_list)

        realworld_list = get_realworld_dataset(csv_fpath=realworld_csv)

        # keep only a subset for development purposes
        if self.small_dataset is True:
            validation_list = validation_list[:1] + training_list[:1]
            training_list = training_list[:1]
            test_list = training_list[:1]
            logger.info("training list {0} validation list {1}".format(training_list, validation_list))
        else:
            logger.info("training list length {0} validation list {1}".format(len(training_list), len(validation_list)))
        # build custom datasets
        self.training_dataset = get_dataset(data_list=training_list, num_shifts=self.num_shifts)
        self.validation_dataset = get_dataset(data_list=validation_list, num_shifts=self.num_shifts)
        self.test_dataset = get_dataset(data_list=test_list, num_shifts=self.num_shifts)
        self.realworld_dataset = get_dataset(data_list=realworld_list, num_shifts=self.num_shifts,
                                             csv_file=realworld_csv, realworld=True)

    def autotune(self, data_dict):
        """ Realworld prediction function"""
        # set up and initialize
        with torch.no_grad():
            batch_size = data_dict['shifts_gt'].size(2)  # the batch size can vary at the last sample of an epoch
            # reset hidden states
            self.model.hidden, self.model.hidden_out = self.model.init_hidden(device=self.device, batch_size=batch_size)
            outputs_list = []  # Store all output notes here
            num_notes = len(data_dict['shifted_pyin'])
            for j in range(num_notes):
                # feedforward
                outputs = self.model(data_dict['spect_v'][j].to(self.device), data_dict['spect_b'][j].to(self.device),
                                     data_dict['spect_c'][j].to(self.device), register_hooks=False,
                                     layer_plot_directory=self.layer_plot_directory)
                outputs_list.append(outputs.detach().cpu().numpy())
        return np.array(outputs_list)

    def eval(self, data_dict, save_song_outputs=False, plot=False):
        """ Evaluation function for one batch"""
        # set up and initialize
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

    def autotune_iters(self, dataloader):
        """ Runs predictions on original singing instead of the synthesized, de-tuned vocal tracks. """
        for j, data_dict in enumerate(dataloader):
            if data_dict is None:
                continue
            outputs = self.autotune(data_dict)
            utils.save_outputs(self.realworld_results_directory, self.realworld_plot_directory, 0,
                    data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                    data_dict['original_boundaries'], pyin_dir=realworld_pyin_directory, training=False,
                    logger=logger)
            utils.synthesize_result(self.realworld_audio_output_directory, data_dict['perf_id'], data_dict['arr_id'],
                                    outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                                    data_dict['original_boundaries'], logger=logger)
            del data_dict; del outputs  # clear memory
        return

    def test_iters(self, epoch, dataloader):
        total_validation_loss = 0.0
        validation_batch_count = 0
        for j, data_dict in enumerate(dataloader):
            print(j)
            if data_dict is None:
                continue
            save_song = True
            plot = False
            outputs, loss = self.eval(data_dict, save_song_outputs=save_song, plot=plot)
            validation_batch_count += 1
            total_validation_loss += loss
            # save sample outputs
            if save_song:
                utils.save_outputs(self.test_results_directory, self.test_plot_directory, epoch,
                        data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                        data_dict['original_boundaries'], training=False, logger=logger)
                utils.synthesize_result(self.test_audio_output_directory, data_dict['perf_id'], data_dict['arr_id'],
                                        outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                                        data_dict['original_boundaries'], logger=logger)
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
                utils.save_outputs(self.training_results_directory, self.training_plot_directory, epoch,
                        data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                        data_dict['original_boundaries'], training=False, logger=logger)
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
                    utils.save_outputs(self.training_results_directory, self.training_plot_directory, epoch,
                            data_dict['perf_id'], outputs, data_dict['shifts_gt'].detach().cpu().numpy(),
                            data_dict['original_boundaries'], training=True, logger=logger)
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
                    bplt.output_file(os.path.join(self.model_plot_directory, "rnn_losses.html"))
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
                    logger.info("is_best before {0} mean loss {1} best prec {2}".format(self.is_best,
                                mean_validation_loss, self.best_prec1))

                    self.is_best = True if mean_validation_loss < self.best_prec1 else False
                    self.best_prec1 = min(mean_validation_loss, self.best_prec1)
                    logger.info("is_best after {0} mean loss {1} best prec {2}".format(self.is_best,
                                mean_validation_loss, self.best_prec1))

                    if self.sandbox is False:
                        save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                         'best_prec1': self.best_prec1, 'optimizer': self.optimizer.state_dict(),
                                         'training_losses': self.training_losses,
                                         'validation_losses': self.validation_losses}, self.is_best,
                                        latest_filename=self.latest_checkpoint_file,
                                        best_filename=self.best_checkpoint_file)
                counter += 1
            logger.info("--- {0} time elapsed for one epoch ---".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN for pitch correction predictions')
    parser.add_argument('--gpu_id', default='0', type=str, help='CUDA visible device')
    parser.add_argument('--epochs', default=100, type=int)
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
    parser.add_argument('--run_autotune', default=False, type=utils.str2bool, help='run autotuner on realworld data')
    parser.add_argument('--use_combination_channel', default=True, type=utils.str2bool,
                        help='combine vocals and backing into third channel')
    parser.add_argument('--extension', default="", help='extension to various files for this experiment')

    args = parser.parse_args()

    logging.basicConfig(filename='log{0}.txt'.format(args.extension), filemode='w',
                        format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.StreamHandler().setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # create and clear directories
    program = Program(extension=args.extension, hidden_size=args.hidden_size, global_dropout=args.global_dropout,
                      gpu_id=args.gpu_id, learning_rate=args.learning_rate, resume=args.resume,
                      small_dataset=args.small_dataset, max_norm=args.max_norm, num_layers=args.num_layers,
                      num_shifts=args.num_shifts, epochs=args.epochs, report_step=args.report_step,
                      sandbox=args.sandbox, use_combination_channel=args.use_combination_channel)
    # run the program
    if args.run_training is True:
        program.train_iters()
    if args.run_testing is True:
        mean_test_loss = program.test_iters(epoch=0, dataloader=program.test_dataset)
        logger.info("Test loss: {0}".format(mean_test_loss))
    if args.run_autotune is True:
        program.autotune_iters(dataloader=program.realworld_dataset)
