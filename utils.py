#!/usr/bin/python2.7

"""Miscellaneous utils for autotune research"""


from __future__ import print_function

__version__ = 2

from globals import *

import argparse
import bokeh.plotting as bplt
import gc
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import psutil
import sys
import torch

import subprocess
import pprint


def buffer(y, frame_length, hop_length):
    """This function is from librosa."""
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def query_performances_from_string(query_str, environment="production"):
    """
    :param query: MK query string
    :param environment: Smule environment
    :return: List of performances
    """
    print('Running query...')
    performance_keys = query.query_performance_keys(None, query_str)
    all_performances = query.performance(performance_keys, environment=environment)
    return all_performances


def get_performances_from_keys(key_list, environment="production"):
    """
    :param key_list: list of performance keys as strings
    :param environment: Smule environment
    :return: List of performances
    """
    all_performances = query.performance(key_list, environment=environment)
    return all_performances


def load_vocals_and_backing_track(perf, fs=global_fs, restrict_range=restrict_range):
    """
    Load audio, remove latency samples, and align solo and back
    :param perf: Performance
    :param fs: sampling rate applied to audio
    :param restrict_range: keep only audio from start_sec to end_sec
    :return: solo and backing track in raw audio format
    """
    solo, fs, fxp = perf.track.vocal.get_audio(channels=1, sampling_rate=global_fs)
    back, fs_back, fxp = perf.song.background.get_audio(channels=1, sampling_rate=global_fs)
    solo = solo[int(perf.track.latency_samples):]
    back = back[int(perf.track.latency_samples):]
    if restrict_range:  # time range of audio written to disk (same as pyin input)
        start, end = int(start_sec * fs), int(end_sec * fs)
        solo = solo[start:end]
        back = back[start:end]
    if len(solo) != len(back):
        min_length = min(len(solo), len(back))
        solo = solo[:min_length]
        back = back[:min_length]
    return solo, back


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
