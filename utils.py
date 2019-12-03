#!/usr/bin/python2.7

"""Miscellaneous utils for autotune research"""

from __future__ import print_function

__version__ = 2


import argparse
import bokeh.plotting as bplt
import calendar
import csv
import gc
from interpolate_pyin import sec_to_frame
import librosa
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pprint
import psutil
import subprocess
import sys
import time
import torch
import traceback

# imports from parallel packages
from globals import *
from psola import psola_shift_pitch
import dataset_analysis


def buffer(y, frame_length, hop_length):
    """This function is from librosa."""
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def reset_directory(directory, empty=False):
    """
    Creates directory if it doesn't exist. Optionally, also removes all pre-existing files
    :param directory: directory path
    :param empty: remove pre-existing files if true
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    if empty is True:
        for item in os.listdir(directory):
            if not os.path.isdir(os.path.join(directory, item)):
                os.remove(os.path.join(directory, item))


def parse_note_csv(fpath):
    with open(fpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        notes = []
        for row in csv_reader:
            start = np.float32(row[0])
            duration = np.float32(row[2])
            notes.append([sec_to_frame(start), sec_to_frame(start + duration)])
    return notes


def clear_cache_brute_force():
    audio_cache_directory = "/tmp/audio_cache"
    for item in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(audio_cache_directory)) for f in fn]:
        if os.path.exists(item):
            os.remove(item)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# https://discuss.pytorch.org/t/how-pytorch-releases-variable-garbage/7277/2?u=zanaa
def memReport():
    """Checks number of tensors and their total size"""

    total_objects = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            # print(type(obj), obj.size())
            total_objects += 1
    print("Total number of tensor objects", total_objects)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def cpuStats():
    """CPU stats to make sure PyTorch isn't hoarding memory at every iteration"""

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def parse_nvidia_smi():
    """
    Parse output of nvidia-smi into a python dictionary.
    This is very basic!
    """

    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].split('\n')  # out_list = out_str[0].decode("utf-8").split('\n')

    out_dict = {}

    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass

    pprint.pprint(out_dict)


def plot_lines(lines, title, layer_plot_directory):
    bplt.output_file(os.path.join(layer_plot_directory, title + ".html"))
    p1 = bplt.figure(title=title + "_1")  # for midi
    p2 = bplt.figure(title=title + "_2")  # for pyin
    for l in lines:
        l = l.detach().cpu().numpy().squeeze()  # get all num_shifts * batch_size arrays
        data1 = l[:, 0]  # example 1
        data2 = l[:, 1]  # example 2
        p1.line(np.arange(len(data1)), data1)  # plot first example in first figure
        p1.xaxis.axis_label = "frames"
        p1.yaxis.axis_label = "frequency"
        p2.line(np.arange(len(data2)), data2)  # plots second example in second figure
        p2.xaxis.axis_label = "frames"
        p2.yaxis.axis_label = "frequency"
        bplt.save(bplt.gridplot([p1], [p2]))


def plot_layer(data, title, dim, layer_plot_directory):
    bplt.output_file(os.path.join(layer_plot_directory, title + ".html"))
    data = data.detach().cpu().numpy()
    if dim == 2:
        data1 = data[:, 0]
        data2 = data[:, 1]
        p1 = bplt.figure(title=title + "_1")
        p1.line(np.arange(len(data1)), data1)
        p1.xaxis.axis_label = "frames"
        p1.yaxis.axis_label = "features range and mean: " + str(np.min(data1)) + ", " + str(np.max(data1)) + \
                              ", " + str(np.mean(data1))
        p2 = bplt.figure(title=title + "_2")
        p2.line(np.arange(len(data2)), data2)
        p2.xaxis.axis_label = "frames"
        p2.yaxis.axis_label = "features range and mean: " + str(np.min(data2)) + ", " + str(np.max(data2)) + \
                              ", " + str(np.mean(data2))
        bplt.save(bplt.gridplot([p1], [p2]))
    if dim == 3:
        data1 = data[:, 0, :]
        data2 = data[:, 1, :]
        p1 = bplt.figure(x_range=(0, data1.shape[0]), y_range=(0, data1.shape[1]), title=title + "_1")
        p1.image(image=[np.log(data1 + np.abs(np.min(data1)) + 1e-10).T], x=[0], y=[0], dw=[data1.shape[0]],
                 dh=[data1.shape[1]], palette='Greys256')
        p1.xaxis.axis_label = "frames"
        p1.yaxis.axis_label = "features range and mean: " + str(np.min(data1)) + ", " + str(np.max(data1)) + \
                              ", " + str(np.mean(data1))
        p2 = bplt.figure(x_range=(0, data2.shape[0]), y_range=(0, data2.shape[1]), title=title + "_2")
        p2.image(image=[np.log(data2 + np.abs(np.min(data2)) + 1e-10).T], x=[0], y=[0], dw=[data2.shape[0]],
                 dh=[data2.shape[1]], palette='Greys256')
        p2.xaxis.axis_label = "frames"
        p2.yaxis.axis_label = "features range and mean: " + str(np.min(data2)) + ", " + str(np.max(data2)) + \
                              ", " + str(np.mean(data2))
        bplt.save(bplt.gridplot([p1], [p2]))


def plot_params(model):
    print("\n\nPlotting parameters listed below: view in", parameter_plot_directory)
    child_counter = 0
    for child in model.children():
        if "Linear" in str(child):
            params = list(child.parameters())
            weights = params[0].detach().cpu().numpy()
            bias = params[1].detach().cpu().numpy()
            bplt.output_file(os.path.join(parameter_plot_directory,
                    "Child_" + str(child_counter) + "_" + str(child) + "weights.html"))
            p = bplt.figure(x_range=(0, weights.shape[0]), y_range=(0, weights.shape[1]))
            p.image(image=[weights], x=[0], y=[0], dw=[weights.shape[0]], dh=[weights.shape[1]],
                    palette='Greys256')
            bplt.save(p)
            bplt.output_file(os.path.join(parameter_plot_directory,
                    "Child_" + str(child_counter) + "_" + str(child) + "bias.html"))
            p = bplt.figure(title="bias")
            p.line(np.arange(len(bias)), bias)
            bplt.save(p)
        elif "GRU" in str(child):
            for name, param in child.named_parameters():
                print(name, param.detach().cpu().numpy().shape)
                if "ih" in str(name) or "hh" in str(name):
                    if "bias" in name:
                        continue
                    w_r, w_i, w_n = param.chunk(3, 0)
                    w_r = w_r.detach().cpu().numpy()
                    w_i = w_i.detach().cpu().numpy()
                    w_n = w_n.detach().cpu().numpy()
                    bplt.output_file(os.path.join(parameter_plot_directory,
                            "Child_" + str(child_counter) + "_GRU_weights.html"))
                    s1 = bplt.figure(x_range=(0, w_r.shape[0]), y_range=(0, w_r.shape[1]), title="w_ir or w_hr")
                    s1.image(image=[w_r], x=[0], y=[0], dw=[w_r.shape[0]], dh=[w_r.shape[1]],
                             palette="Greys256")
                    s2 = bplt.figure(x_range=(0, w_i.shape[0]), y_range=(0, w_i.shape[1]), title="w_ii or w_hi")
                    s2.image(image=[w_i], x=[0], y=[0], dw=[w_i.shape[0]], dh=[w_i.shape[1]],
                             palette="Greys256")
                    s3 = bplt.figure(x_range=(0, w_n.shape[0]), y_range=(0, w_n.shape[1]), title="w_in or w_hn")
                    s3.image(image=[w_n], x=[0], y=[0], dw=[w_n.shape[0]], dh=[w_n.shape[1]],
                             palette="Greys256")
                    p = bplt.gridplot([[s1, s2, s3]], toolbar_location=None)
                    bplt.save(p)
                else:
                    pass  # TODO plot the bias
        elif "SqueezeNetModified" in str(child):
            for name, param in child.named_parameters():
                if "weight" in name and not "expand" in name:
                    print("weight in name", name)
                    weight = param.detach().cpu().numpy()
                    print("size:", weight.shape)
                    bplt.output_file(os.path.join(parameter_plot_directory, name + ".html"))
                    s1 = bplt.figure(x_range=(0, weight.shape[0]), y_range=(0, weight.shape[1]), title="w_ir or w_hr")
                    s1.image(image=[weight], x=[0], y=[0], dw=[weight.shape[0]], dh=[weight.shape[1]],
                             palette="Greys256")
                    bplt.save(s1)
                else:
                    pass
        elif "Conv2d" in str(child):
            for name, param in child.named_parameters():
                if "weight" in name and not "expand" in name:
                    print("Conv2d weight in name", name)
                    weight = param.detach().cpu().numpy()
                    print("size:", weight.shape)
                    bplt.output_file(os.path.join(parameter_plot_directory, name + ".html"))
                    s1 = bplt.figure(x_range=(0, weight.shape[0]), y_range=(0, weight.shape[1]), title="w_ir or w_hr")
                    s1.image(image=[weight], x=[0], y=[0], dw=[weight.shape[0]], dh=[weight.shape[1]],
                             palette="Greys256")
                    bplt.save(s1)
                else:
                    pass  # TODO plot the bias
        else:
            print("Not plotting", str(child))
        child_counter += 1


def print_param_sizes(model):
    print("\n\nModel parameters, sizes, and freeze status:")
    child_counter = 0
    for child in model.children():
        print("Child", child_counter, "is:", child)
        for param in child.parameters():
            print("requires grad:", param.requires_grad == True)
            data = param.detach().cpu().numpy()
            if np.isscalar(data):
                print(data)
            else:
                print(data.shape)
        child_counter += 1


def print_params(model):
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is:", child)
        for param in child.parameters():
            data = param.detach().cpu().numpy()
            if isinstance(data, int) or np.isscalar(data):
                continue
            else:
                print("size", data.shape)
                if np.isnan(np.sum(data)):
                    print("Warning: parameter contains Nan")
                print("min", np.min(data),
                      "max", np.max(data),
                      "mean", np.mean(data))
        child_counter += 1


def save_outputs(results_numpy_dir, results_plot_dir, epoch, perf, outputs, labels_tensor, original_boundaries,
                 training, logger, pyin_dir=pyin_directory):
    """Save predictions and labels for one shift of every song in batch"""
    try:
        f_start = "training" if training is True else "testing"
        ts = str(calendar.timegm(time.gmtime()))
        bplt.output_file(os.path.join(
                results_plot_dir, f_start + "_epoch_" + str(epoch) + "_" + ts + "_" + perf + ".html"))

        # load the original, in-tune pitch track
        pitch_track = np.load(os.path.join(pyin_dir, perf + ".npy"))
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
            results_numpy_dir, f_start + "_epoch_" + str(epoch) + "_outputs" + "_" + perf), outputs)
        np.save(os.path.join(
            results_numpy_dir, f_start + "_epoch_" + str(epoch) + "_labels" + "_" + perf), labels)
    except Exception as e:
        tb = traceback.format_exc()
        logger.info("exception in save_outputs {0}: {1} skipping song {2}".format(e, tb, perf))
        return


def synthesize_result(output_dir, perf, arr, outputs, labels_tensor, original_boundaries, logger,
                      pyin_dir=pyin_directory):
    test_mix_fpath = os.path.join(output_dir, "test_mix_" + perf + ".wav")
    shifted_mix_fpath = os.path.join(output_dir, "shifted_mix_" + perf + ".wav")
    corrected_mix_fpath = os.path.join(output_dir, "corrected_mix_" + perf + ".wav")
    # load the original, in-tune pitch track
    pitch_track = np.load(os.path.join(pyin_dir, perf + ".npy"))
    frames = len(pitch_track)

    # load the original audio and write mixture to file
    vocal_audio = dataset_analysis.get_audio(os.path.join(vocals_directory, perf + ".wav"))
    backing_audio = dataset_analysis.get_audio(os.path.join(backing_tracks_directory, arr + ".wav"), use_librosa=True)
    if arr == 'silent_backing_track':
        logger.info('using silent backing track')
        backing_audio *= 0

    min_len = min([len(backing_audio), len(vocal_audio)])
    test_mix = vocal_audio[:min_len] + backing_audio[:min_len]  # check that the audio is correct
    # test_mix = test_mix[:30 * global_fs]
    librosa.output.write_wav(test_mix_fpath, test_mix, sr=global_fs, norm=True)

    # convert autotuner outputs to shape < notes, shifts >
    outputs = np.squeeze(outputs)
    labels = np.squeeze(labels_tensor)
    correction_shifts = labels - outputs
    labels_ratios = np.power(2, labels[:, 0] / 12)
    corrections_ratios = np.power(2, correction_shifts[:, 0] / 12)

    # plot the shifts after applying them to the original frame indices
    frame_outputs_0 = np.zeros(frames)
    frame_labels_0 = np.zeros(frames)
    frame_corrections_0 = np.zeros(frames)
    for i in range(len(labels[:, 0])):
        frame_outputs_0[original_boundaries[i, 0]: original_boundaries[i, 1]] += outputs[i, 0]
        frame_labels_0[original_boundaries[i, 0]: original_boundaries[i, 1]] += labels[i, 0]
        frame_corrections_0[original_boundaries[i, 0]: original_boundaries[i, 1]] -= correction_shifts[i, 0]
    s1 = bplt.figure(title="Pitch shifts: ground truth, predictions, and shift (ground truth is zero if autotuning")
    s1.line(np.arange(len(frame_labels_0)), frame_labels_0, color="green")
    s1.line(np.arange(len(frame_outputs_0)), frame_outputs_0, color="blue")
    s1.line(np.arange(len(frame_corrections_0)), frame_corrections_0, color="orange")
    bplt.save(s1, filename=os.path.join(output_dir, "plot_" + perf + ".html"))

    # shift pitch to get the de-tuned input to the neural net, then apply correction (negative of learned shift)
    shifted_pitch_track_0 = np.copy(pitch_track)
    corrected_pitch_track_0 = np.copy(pitch_track)
    for i in range(len(labels[:, 0])):
        shifted_pitch_track_0[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                    np.power(2, max_semitone * labels[i, 0] / 12.0)
        corrected_pitch_track_0[original_boundaries[i, 0]: original_boundaries[i, 1]] *= \
                    np.power(2, max_semitone * (labels[i, 0] - outputs[i, 0]) / 12.0)
    s1 = bplt.figure(title="Pitch tracks: ground truth, predictions, and shift (ground truth is zero if autotuning")
    s1.line(np.arange(len(shifted_pitch_track_0)), shifted_pitch_track_0, color="green")
    s1.line(np.arange(len(corrected_pitch_track_0)), corrected_pitch_track_0, color="orange")
    bplt.save(s1, filename=os.path.join(output_dir, "plot_pyin" + perf + ".html"))

    # do the same with the audio
    shifted_audio = np.copy(vocal_audio)
    corrected_audio = np.copy(vocal_audio)
    margin = 1
    for i in range(len(labels[:, 0])):
        if i == 0:
            start_sample = max((original_boundaries[i, 0] - margin) * hopSize, 0)
        else:
            start_sample = max((original_boundaries[i, 0] - margin) * hopSize, original_boundaries[i - 1, 1] * hopSize)
        if i == len(original_boundaries) - 1:
            end_sample = min((original_boundaries[i, 1] + margin) * hopSize, len(vocal_audio))
        else:
            end_sample = min((original_boundaries[i, 1] + margin) * hopSize, original_boundaries[i + 1, 0] * hopSize)
        start_sample = int(start_sample)
        end_sample = int(end_sample)
        # get shifted audio
        temp_shifted = psola_shift_pitch(
            vocal_audio[start_sample: end_sample], fs=global_fs, f_ratio_list=[labels_ratios[i]])[0]
        temp_shifted[0: margin * hopSize] *= np.linspace(0, 1, margin * hopSize)
        temp_shifted[-margin * hopSize: -1] *= np.linspace(1, 0, margin * hopSize - 1)
        shifted_audio[start_sample: start_sample + margin * hopSize] *= np.linspace(1, 0, margin * hopSize)
        shifted_audio[end_sample - margin * hopSize: end_sample] *= np.linspace(0, 1, margin * hopSize)
        shifted_audio[start_sample + margin * hopSize: end_sample - margin * hopSize] = 0
        shifted_audio[start_sample: end_sample] += temp_shifted
        # get corrected audio
        temp_corrected = psola_shift_pitch(
            vocal_audio[start_sample: end_sample], fs=global_fs, f_ratio_list=[corrections_ratios[i]])[0]
        temp_corrected[0: margin * hopSize] *= np.linspace(0, 1, margin * hopSize)
        temp_corrected[-margin * hopSize: -1] *= np.linspace(1, 0, margin * hopSize - 1)
        corrected_audio[start_sample: start_sample + margin * hopSize] *= np.linspace(1, 0, margin * hopSize)
        corrected_audio[end_sample - margin * hopSize: end_sample] *= np.linspace(0, 1, margin * hopSize)
        corrected_audio[start_sample + margin * hopSize: end_sample - margin * hopSize] = 0
        corrected_audio[start_sample: end_sample] += temp_corrected

    min_len = min([len(backing_audio), len(shifted_audio)])
    shifted_mix = np.array(shifted_audio[:min_len]) + backing_audio[:min_len]
    corrected_mix = np.array(corrected_audio[:min_len]) + backing_audio[:min_len]

    librosa.output.write_wav(shifted_mix_fpath, shifted_mix, sr=global_fs, norm=True)
    librosa.output.write_wav(corrected_mix_fpath, corrected_mix, sr=global_fs, norm=True)
