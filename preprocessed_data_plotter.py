#!/usr/bin/python3.6


"""Reproduce Dataloader functionality in rnn.py while plotting the data"""


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# other imports from parallel packages
import dataset_analysis
from globals import *
import utils

import argparse
import bokeh.plotting as bplt
from bokeh.models import Span
from skimage.filters import threshold_mean
import csv
import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)


class AutotuneDataset(Dataset):
    """Customized PyTorch Dataset that treats one song with all its pitch-shifted versions as a data sample.
    This function is used as input to the DataLoader."""

    def __init__(self, performance_list, num_shifts, metadata_csv, plot=False, freeze=False):
        self.performance_list = performance_list
        self.plot = plot
        self.num_shifts = num_shifts
        self.freeze = freeze
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

    def __getitem__(self, idx):
        """
        :param keep: number of songs to keep, less or equal to the number of pitch shifts
        :return: a list of length num_performance_shifts.
        Each item is a tuple with the labels <seq len, shifts>, input <seq len, shifts, fdim>, key (str)
        """
        fpath = os.path.join(autotune_preprocessed_directory, self.performance_list[idx] + ".pkl")
        if not os.path.exists(fpath):
            try:
                # pass if acctID is 33128648
                pyin = np.load(os.path.join(pyin_directory, self.performance_list[idx] + ".npy"))
                # load stft of vocals, keep complex values in order to use istft later for pitch shifting
                stft_v = dataset_analysis.get_stft(
                    os.path.join(vocals_directory, self.performance_list[idx] + ".wav")).T
                # load cqt of backing track
                cqt_b = np.abs(dataset_analysis.get_cqt(os.path.join(backing_tracks_directory,
                                                                     self.arr_keys[
                                                                         self.performance_list[idx]] + ".wav"))).T
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
                        loc = Span(location=ab + 1, dimension='height', line_color='red', line_dash='dotted')
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

                start_f = 0
                end_f = 300
                matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True)
                ax1.imshow(np.log(cqt_b[start_f:end_f, 0, :].T + 1e-10), aspect='auto', origin='lower')
                ax1.set_ylabel("CQT bins")
                ax1.set_title("Backing track CQT")
                ax1.set_xlabel("frames")
                ax4.plot(pyin[start_f:end_f] * 500 / np.max(pyin[start_f:end_f]))
                for _, (beg, end) in enumerate(notes):
                    if beg >= start_f and beg <= end_f:
                        ax4.axvline(x=beg, color="green")
                    if end >= start_f and end <= end_f:
                        ax4.axvline(x=end, ls='dotted', color="red")
                ax4.set_xlabel("frames")
                ax4.set_title("Note parse")
                ax2.imshow(np.log(stacked_cqt_v[start_f:end_f, 0, :].T + 1e-10), aspect='auto', origin='lower')
                ax2.set_title("Vocals CQT (in tune)")
                ax2.set_xlabel("frames")
                ax5.imshow(stacked_cqt_combined[start_f:end_f, 0, :].T, aspect='auto', origin='lower')
                ax5.set_title("Difference (in tune)")
                ax5.set_xlabel("frames")
                ax3.imshow(np.log(stacked_cqt_v[start_f:end_f, 2, :].T + 1e-10), aspect='auto', origin='lower')
                ax3.set_title("Vocals CQT (de-tuned)")
                ax3.set_xlabel("frames")
                ax6.imshow(stacked_cqt_combined[start_f:end_f, 2, :].T, aspect='auto', origin='lower')
                ax6.set_title("Difference (de-tuned)")
                ax6.set_xlabel("frames")
                plt.tight_layout()
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_3_" +
                            self.performance_list[idx] + ".eps", format="eps")
                plt.show()

                matplotlib.rcParams.update({'font.size': 20})
                plt.imshow(np.log(cqt_b[start_f:end_f, 0, :].T + 1e-10), aspect=0.6, origin='lower')
                plt.ylabel("CQT bins")
                plt.xlabel("frames")
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_1.eps", format="eps",
                            bbox_inches='tight')
                plt.clf()
                plt.gca()
                plt.cla()

                plt.imshow(np.log(stacked_cqt_b_binary[start_f:end_f, 0, :].T + 1e-10), aspect=0.6, origin='lower')
                plt.xlabel("frames")
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_2.eps", format="eps",
                            bbox_inches='tight')
                plt.clf()
                plt.gca()
                plt.cla()

                plt.imshow(np.log(stacked_cqt_v[start_f:end_f, 0, :].T + 1e-10), aspect=0.6, origin='lower')
                plt.xlabel("frames")
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_3.eps", format="eps",
                            bbox_inches='tight')
                plt.clf()
                plt.gca()
                plt.cla()

                plt.imshow(stacked_cqt_combined[start_f:end_f, 0, :].T, aspect=0.6, origin='lower')
                plt.xlabel("frames")
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_4.eps", format="eps",
                            bbox_inches='tight')
                plt.clf()
                plt.gca()
                plt.cla()

                plt.imshow(np.log(stacked_cqt_v[start_f:end_f, 2, :].T + 1e-10), aspect=0.6, origin='lower')
                plt.xlabel("frames")
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_5.eps", format="eps",
                            bbox_inches='tight')
                plt.clf()
                plt.gca()
                plt.cla()

                plt.imshow(stacked_cqt_combined[start_f:end_f, 2, :].T, aspect=0.6, origin='lower')
                plt.xlabel("frames")
                frame1 = plt.gca()
                frame1.axes.yaxis.set_ticklabels([])
                plt.savefig("/Users/scwager/Documents/autotune_fa18_data/plots/cqt_comparison_6.eps", format="eps",
                            bbox_inches='tight')
                plt.show()
                plt.clf()
                plt.gca()
                plt.cla()
                # ---------------------------------

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


def get_dataset(data_list, songs_per_batch, num_shifts, device, mode="testing", workers_on_gpu=2, freeze=False):
    dataset = AutotuneDataset(data_list, num_shifts=num_shifts, metadata_csv=metadata_csv, plot=False, freeze=freeze)
    return dataset


def split_into_training_validation_test(performance_list):
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
    validation_list = [p for _, p in enumerate(performance_list) if arr_keys[p] <= "3769590_3769590"]
    test_list = validation_list
    training_list = sorted(list(set(performance_list) - set(validation_list)))
    return training_list, validation_list, test_list


class Program:
    def __init__(self, num_shifts):
        self.num_shifts = num_shifts

        utils.reset_directory(autotune_preprocessed_directory, empty=False)
        logger.info("Results will be saved in: {0}".format(autotune_preprocessed_directory))

        # load the performance indices for the datasets: all ids that have an instance in all directories
        performance_list = sorted(list(
            set([f[:-4] for f in os.listdir(pyin_directory) if "npy"]) &
            set([f[:-4] for f in os.listdir(midi_directory) if "npy"]) &
            set([f[:-4] for f in os.listdir(back_chroma_directory) if "npy"])))

        training_list, validation_list, test_list = split_into_training_validation_test(performance_list)

        print("training list", len(training_list), training_list[:5], training_list[-5:])
        training_list = training_list[2500:] + training_list[:2500]
        print("rotated training list", len(training_list), training_list[:5], training_list[-5:])
        logger.info("training list {0} validation list {1}".format(len(training_list), len(validation_list)))
        # build custom datasets
        freeze = False
        self.device = ("cpu")
        logger.info("fixing some of the shifts across all songs: {0}".format(freeze))
        self.training_dataset = get_dataset(data_list=training_list, num_shifts=self.num_shifts,
                                            songs_per_batch=1, mode="training", device=self.device, freeze=freeze)
        self.validation_dataset = get_dataset(data_list=validation_list, num_shifts=self.num_shifts,
                                              songs_per_batch=1, mode="testing", device=self.device, freeze=freeze)
        self.test_dataset = get_dataset(data_list=test_list, num_shifts=self.num_shifts,
                                        songs_per_batch=1, mode="testing", device=self.device, freeze=freeze)

    def plot_data(self):
        counter = 0
        # training
        for i, data_dict in enumerate(self.training_dataset):
            if i < 2 or i % 30 == 0 or counter < 0 or counter % 30 == 0:
                logger.info("step {0} counter {1}".format(i, counter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program for plotting the input data')

    parser.add_argument('--num_shifts', default=7, type=int, help='number of shifts per song')
    parser.add_argument('--extension', default="plotter", help='log file extension')

    args = parser.parse_args()

    logging.basicConfig(filename='log{0}.txt'.format(args.extension), filemode='w',
                        format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.StreamHandler().setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # create and clear directories
    program = Program(num_shifts=args.num_shifts)
    # run the program
    program.plot_data()
