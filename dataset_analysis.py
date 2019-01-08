
"""
Program for
- Audio loading and pre-processing (e.q. Constant-Q Transform)
Optional MIDI analysis (if MIDI score is available, but this is not required for rnn.py)
- Alignment of pitch track and MIDI using Dynamic Time Warping
- Shifting MIDI to the nearest octave
- Plots such as a global histogram of pitch deviations in a dataset, used to create
plots in

S. Wager, G. Tzanetakis, C. Wang, S. Sullivan, J. Shimmin, M. Kim, and P. Cook,
“Intonation: A dataset of quality vocal performances refined by spectral clustering on pitch congruence,”
in IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Submitted for publication.
Available at http://homes.sice.indiana.edu/scwager/images/damp_dataset_nov5.pdf
"""


from globals import *
import utils

import argparse
from collections import Counter
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa
import numpy as np
import os
import pickle


base_directory = os.environ.get("DATA_ROOT", "/share/project/scwager/autotune_fa18_data/")
clustering_pitch_dir = "clustering_data_sanna/vocals_pitch_pyin"
intonation_pitch_dir = "Intonation/vocals_pitch_pyin"
clustering_midi_dir = "clustering_data_sanna/vocals_midi"
intonation_midi_dir = "Intonation/vocals_midi"
analysis_dir = "analysis"
differences_dir = os.path.join(base_directory, analysis_dir, "differences")


def adjust_octave(midi_hz, measured_hz):
    """
    Adjusts midi to the nearest octave in case of octave error.
    :param midi_hz: numpy array of frame-wise MIDI frequencies in Hz
    :param measured_hz: numpy array of frame-wise performance pitch track frequencies in Hz
    :param performance: Performance instance
    :return: array of differences in cents and the octave-adjusted midi_hz
    """
    # find non-silent frames
    singing_region = np.where((measured_hz > 1.0) & (midi_hz > 1.0))[0]
    cent_differences = np.log2((midi_hz + 1e-10) / (measured_hz + 1e-10)) * 1200
    octaves = np.arange(-3, 4) * 1200
    octave_error = octaves[np.argmin(np.abs(octaves - np.median(cent_differences[singing_region])))]
    midi_hz = np.power(2, (cent_differences - octave_error) / 1200) * measured_hz
    return midi_hz


def dtw(midi_hz, measured_hz, performance_key, plot=False, plot_dir="./"):
    _, wp = librosa.sequence.dtw(midi_hz, measured_hz, step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0], [2, 0]]),
            weights_add=np.array([0, 100, 0, 0]), weights_mul=np.array([1, 10, 1, 1]))
    if wp[0, 0] > wp[-1, 0]:
        wp = np.flip(wp, axis=0)
    midi_hz = midi_hz[wp[:, 0]]
    measured_hz = measured_hz[wp[:, 1]]
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Frames")
        ax.set_ylabel("Frequency (Hz)")
        plt.plot(midi_hz[:1500], color="green", label="MIDI")
        plt.plot(measured_hz[:1500], color="purple", label="pYIN")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "midi_and_pyin_aligned_" + performance_key + ".eps"), format="eps")
        plt.close()
    return midi_hz, measured_hz, wp


def get_audio(filepath):
    """
    Reads audio from disk and normalize
    :param filepath: pitch shifted wav file path
    :return:
    """

    audio, fs = librosa.load(path=filepath, sr=22050)
    latency_samples = int(fs / 1000 * 10)  # assume latency is 10 ms, typical for iOS
    latency_samples *= 2  # assume the audio was treated differently when put into the Intonation dataset than
    # when processed at Smule
    # retrieve audio and adjust for latency
    audio = audio[latency_samples:]
    # frame-wise calculation
    if restrict_range:
        start = start_sec * fs
        end = end_sec * fs
        audio = np.array(audio[start:end], dtype=np.float32)
    # normalize
    audio = (cqt_params['normalizing_constant'] * audio) / np.std(audio[np.abs(audio > 0.00001)])
    return audio


def get_cqt(filepath):
    """
    Computes the STFT of the de-tuned audio, read from disk
    :param filepath: pitch shifted wav file path
    :return: CQT
    """

    audio, fs = librosa.load(path=filepath, sr=22050)
    latency_samples = int(fs / 1000 * 10)  # assume latency is 10 ms, typical for iOS
    latency_samples *= 2  # assume the audio was treated differently when put into the Intonation dataset than
    # when processed at Smule
    # retrieve audio and adjust for latency
    audio = audio[latency_samples:]
    # frame-wise calculation
    if restrict_range:
        start = start_sec * fs
        end = end_sec * fs
        audio = np.array(audio[start:end], dtype=np.float32)
    # normalize
    audio = (cqt_params['normalizing_constant'] * audio) / np.std(audio[np.abs(audio > 0.00001)])
    # normalize
    cqt = np.abs(librosa.core.cqt(audio, fmin=cqt_params['fmin'], sr=global_fs, hop_length=hopSize,
                                  n_bins=cqt_params['total_bins'], bins_per_octave=cqt_params['bins_per_8va']))
    return cqt


def get_stft(filepath):
    """
    Computes the STFT of the de-tuned audio, read from disk
    :param filepath: pitch shifted wav file path
    :return: CQT
    """
    audio, fs = librosa.load(path=filepath, sr=22050)
    latency_samples = int(fs / 1000 * 10)  # assume latency is 10 ms, typical for iOS
    latency_samples *= 2  # assume the audio was treated differently when put into the Intonation dataset than
    # when processed at Smule
    # retrieve audio and adjust for latency
    audio = audio[latency_samples:]
    # frame-wise calculation
    if restrict_range:
        start = start_sec * fs
        end = end_sec * fs
        audio = np.array(audio[start:end], dtype=np.float32)
    # normalize
    audio = (cqt_params['normalizing_constant'] * audio) / np.std(audio[audio != 0])
    # compute performance pitch track
    stft = librosa.stft(audio, n_fft=frameSize, hop_length=hopSize, center=False)
    return stft


def get_midi_to_performance_difference(midi_hz, measured_hz, performance_key, plot=False, plot_dir="./"):
    """
    Computes frame-wise differences in cents between the midi score and measured f0 when both are non-silent.
    :param midi_hz: numpy array of frame-wise MIDI frequencies in Hz
    :param measured_hz: numpy array of frame-wise performance pitch track frequencies in Hz
    :param performance_key: Performance identifier
    :return: array of differences in cents when both arrays are non-zero (not silent)
    """
    # find non-silent frames
    singing_region = np.where((measured_hz > 1.0) & (midi_hz > 1.0))[0]
    # compute the differences in cents
    cent_differences = np.log2((midi_hz[singing_region] + 1e-10) / (measured_hz[singing_region] + 1e-10)) * 1200
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Frames")
        ax.set_ylabel("Cents")
        ax.set_title("Difference between MIDI and pYIN in non-silent frames")
        plt.plot(cent_differences)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "midi_performance_diff_cents_" + performance_key + ".eps"), format="eps")
        plt.close()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Frames")
        ax.set_ylabel("Scientific pitch")
        note_names = ['Gb4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 'C5', 'Db5', 'D5', 'Eb5', 'E5', 'F5', 'Gb5', 'G5',
                        'Ab5', 'A5', 'Bb5', 'B5', 'C6']
        ax.set_ylim(np.log2(370), np.log2(940))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.log2(440 * np.power(2, np.arange(-3, 14)/12))))
        ax.set_yticklabels(note_names, fontsize='small')
        for label in (ax.get_xticklabels()):
            label.set_fontsize('small')
        plt.plot(np.log2(midi_hz[singing_region][500:1500]), color="green", label="MIDI")
        plt.plot(np.log2(measured_hz[singing_region][500:1500]), color="purple", label="pYIN")
        plt.legend(loc=4, fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "midi_and_pyin_aligned_active_" + performance_key + ".eps"), format="eps")
        plt.close()
    return cent_differences


def get_histogram(pitch_dir, midi_dir, plots_dir, differences_dir, max_count):
    counter = 0
    histogram = Counter()
    for fname in os.listdir(os.path.join(base_directory, pitch_dir)):
        try:
            if not fname.endswith(".npy"):
                continue
            if counter >= max_count:
                continue
            difference_path = os.path.join(differences_dir, fname)
            if os.path.exists(difference_path):
                differences = np.load(difference_path)
            else:
                pitch_path = os.path.join(base_directory, pitch_dir, fname)
                midi_path = os.path.join(base_directory, midi_dir, fname)
                # load the files
                pitch = np.load(pitch_path)
                midi = np.load(midi_path)
                # truncate to the same length
                min_length = min(len(pitch), len(midi))
                pitch = pitch[:min_length]
                midi = midi[:min_length]
                # shift MIDI by global constant to the octave closest to the pYIN track
                midi = adjust_octave(midi, pitch)
                # align the two by modifying both according to dtw result
                midi, pitch, _ = dtw(midi, pitch, fname[:-4], plot=False, plot_dir=plots_dir)
                # compute difference of all frames where both tracks are non-silent
                differences = get_midi_to_performance_difference(
                        midi, pitch, fname[:-4], plot=True, plot_dir=plots_dir).astype(int)
                np.save(difference_path, differences)
            # add to histogram
            counts = Counter(differences)
            histogram += counts
            counter += 1
        except FileNotFoundError as err:
            print(err)
            continue
        except Exception as e:
            print("get_historgams:", e)
            continue
    return histogram


def bootstrap(midi_dir, differences_dir, cents_min, cents_max):
    # load all the data
    bootstrap_samples = 10000
    dataset = []
    for fname in os.listdir(os.path.join(base_directory, midi_dir)):
        difference_path = os.path.join(differences_dir, fname)
        if not os.path.exists(difference_path) or not fname.endswith(".npy"):
            continue
        differences = np.load(difference_path)
        clustering_classes = np.zeros(2)
        clustering_classes[0] += np.sum((differences > cents_min) & (differences <= cents_max))
        clustering_classes[1] += np.sum((differences >= -cents_max) & (differences < -cents_min))
        dataset.append(clustering_classes)
    dataset = np.array(dataset)

    n = len(dataset)
    print("dataset length", n)
    p_positive = np.zeros(bootstrap_samples)
    for i in range(bootstrap_samples):
        indices = np.random.randint(0, n, n)
        sums = np.sum(dataset[indices], axis=0)
        sums /= np.sum(sums)
        p_positive[i] = sums[0]
    print("class diff", p_positive)
    print('mean', np.mean(p_positive), 'var', np.sqrt(np.var(p_positive)))
    return p_positive


def plot_pipeline(differences_dir, plots_dir_intonation, num_quantiles = 31):
    """
    Samples each song's midi-to-performance difference array to produce fixed-length arrays.
    Computes quantiles of the resulting distribution
    :param comparisons_list: list of arrays of differences (in cents) between midi and performance
    :return: list of quantiles for each song
    """
    perf_list = ['54363310_1939750539', '540791114_1793842568']
    difference_path_list = [os.path.join(differences_dir, perf_list[i] + ".npy") for i in range(len(perf_list))]
    comparisons_list = [np.load(path) for _, path in enumerate(difference_path_list)]
    num_samples = 10000
    # quantile indices
    q_indices = (np.linspace(0, 1, num_quantiles)*(num_samples-1)).astype(np.int32)
    plt.style.use('ggplot')
    labels = ['perf. A', 'perf. B']
    colors = ['blue', 'red']
    linestyles = ['dotted', 'dashed']
    grid = plt.GridSpec(2, 2)
    ax1 = plt.subplot(grid[1, 0])
    ax2 = plt.subplot(grid[1, 1])
    ax4 = plt.subplot(grid[0, :])
    ax4.plot(comparisons_list[0], color=colors[0], label=labels[0], linestyle=linestyles[0])
    ax4.plot(comparisons_list[1], color=colors[1], label=labels[1], linestyle=linestyles[1])
    ax4.set_title("Difference between MIDI and pYIN, two performances")
    ax4.set_ylabel("Cents")
    ax4.set_xlabel("Frames")
    ax4.axhline(y=200, linestyle="solid", linewidth=0.7, c="black", zorder=2, label="thresh.")
    ax4.axhline(y=-200, linestyle="solid", linewidth=0.7, c="black", zorder=2)
    ax4.legend(loc="upper right")
    ax1.set_title("10k random sample of distances")
    ax1.set_ylabel(r"$|$Cents$|$")
    ax1.set_xlabel("Frames sorted by distance")
    ax2.set_title("Sample quantiles")
    ax2.set_xlabel("Quantile indices")
    # run analysis song by song
    for i, arr in enumerate(comparisons_list):
        # random sample so all arrays have the same size
        samples = np.random.choice(arr, num_samples, replace=True)
        # sort
        samples = np.sort(np.abs(samples))
        # discard the high values (might be due to misalignment, etc...)
        samples = samples[samples <= 200]
        samples = np.random.choice(samples, num_samples, replace=True)
        samples = np.sort(np.abs(samples))
        ax1.plot(samples, color=colors[i], linestyle=linestyles[i], label=labels[i])
        # get the quantiles
        samples = samples[q_indices]
        ax2.plot(samples, color=colors[i], linestyle=linestyles[i], label=labels[i])
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir_intonation, "data processing pipeline.eps"), format="eps")
    plt.show()


def main(args):
    utils.reset_directory(os.path.join(base_directory, analysis_dir))
    utils.reset_directory(differences_dir)
    intonation_hist_path = os.path.join(base_directory, analysis_dir, "intonation_hist.pkl")
    clustering_hist_path = os.path.join(base_directory, analysis_dir, "clustering_hist.pkl")

    plots_dir_intonation = os.path.join(base_directory, "plots/Intonation/")
    plots_dir_clustering = os.path.join(base_directory, "plots/clustering_data_sanna/")

    utils.reset_directory(plots_dir_intonation)
    utils.reset_directory(plots_dir_clustering)

    plt.style.use('classic')
    if args.get_histogram is True:
        # load the files if they exist
        if os.path.exists(intonation_hist_path):
            with open(intonation_hist_path, "rb") as fname:
                intonation_hist = pickle.load(fname)
            with open(clustering_hist_path, "rb") as fname:
                clustering_hist = pickle.load(fname)
        # otherwise, run the analysis
        else:
            print("computing histograms...")
            intonation_hist = get_histogram(intonation_pitch_dir, intonation_midi_dir,
                    plots_dir_intonation, differences_dir, args.max_count)
            clustering_hist = get_histogram(clustering_pitch_dir, clustering_midi_dir,
                    plots_dir_clustering, differences_dir, args.max_count)
            with open(os.path.join(base_directory, analysis_dir, "intonation_hist.pkl"), "wb") as fname:
                pickle.dump(intonation_hist, fname)
            with open(os.path.join(base_directory, analysis_dir, "clustering_hist.pkl"), "wb") as fname:
                pickle.dump(clustering_hist, fname)
        # process and normalize
        print("clustering", clustering_hist.most_common(10))
        print("intonation", intonation_hist.most_common(10))
        intonation_hist = np.array(list(map(list, zip(*sorted(intonation_hist.items())))))
        clustering_hist = np.array(list(map(list, zip(*sorted(clustering_hist.items())))))
        print("sums", np.sum(intonation_hist[1]), np.sum(clustering_hist[1]), )
        normalization = np.sum(intonation_hist[1])/np.sum(clustering_hist[1])
        clustering_hist[1] = (clustering_hist[1] * normalization).astype(int)

        # plot full histograms comparison
        # linear scale
        fig = plt.figure(figsize=(8, 5))
        plt.plot(clustering_hist[0], (clustering_hist[1]),
                label="Remaining clusters", color="red", linestyle="dotted")
        plt.plot(intonation_hist[0], (intonation_hist[1]),
                label="Selected clusters", color="blue", linewidth=0.75, linestyle="solid")
        plt.xlabel("Deviations (cents)", fontsize='large')
        plt.ylabel("Occurrences in 1000s", fontsize='large')
        plt.xlim(-1600, 1600)
        plt.ylim(0, 213000)
        ax = plt.axes()
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        # log scale
        ax2 = ax.twinx()
        ax2.set_ylabel("Log of occurrences", fontsize='large')
        ax2.set_ylim(0, 40)
        ax2.yaxis.label.set_size(18)
        ax2.plot(clustering_hist[0], np.log(clustering_hist[1]+1),
                label="Remaining (log)", color="orange", linestyle="dotted")
        ax2.plot(intonation_hist[0], np.log(intonation_hist[1]+1),
                label="Selected (log)", color="green", linewidth=0.75, linestyle="solid")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir_intonation, "full_histograms_comparison.eps"), format="eps")
        fig.show()

        plt.style.use('ggplot')
        # plot full histograms comparison
        # linear scale
        plt.plot(clustering_hist[0], (clustering_hist[1]),
                 label="Remaining clusters", color="red", linewidth=0.75)
        plt.plot(intonation_hist[0], (intonation_hist[1]),
                 label="Selected clusters", color="blue", linewidth=0.75)
        plt.xlabel("Deviations (cents)")
        plt.ylabel("Occurrences in 1000s")
        plt.xlim(-500, 500)
        plt.ylim(0, 25000)
        ax = plt.axes()
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        # log scale
        ax2 = ax.twinx()
        ax2.set_ylabel("Log of occurrences")
        ax2.set_ylim(0, 40)
        ax2.plot(clustering_hist[0], np.log(clustering_hist[1] + 1),
                 label="Remaining clusters (log)", color="orange", linewidth=0.75)
        ax2.plot(intonation_hist[0], np.log(intonation_hist[1] + 1),
                 label="Selected clusters (log)", color="green", linewidth=0.75)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir_intonation, "full_histograms_comparison_zoom.eps"), format="eps")
        plt.show()

        rng = 100
        ind_pos = np.arange(rng + 1)  # the x locations for the groups
        width = 0.35  # the width of the bars

        intonation_pos = np.zeros(rng + 1)
        intonation_neg = np.zeros(rng + 1)
        for i in range(1, rng + 1):
            if i in intonation_hist[0]:
                intonation_pos[i] = intonation_hist[1][np.where(intonation_hist[0] == i)[0][0]]
            if -i in clustering_hist[0]:
                intonation_neg[i] = intonation_hist[1][np.where(intonation_hist[0] == -i)[0][0]]

        clustering_pos = np.zeros(rng + 1)
        clustering_neg = np.zeros(rng + 1)
        for i in range(1, rng + 1):
            if i in clustering_hist[0]:
                clustering_pos[i] = clustering_hist[1][np.where(clustering_hist[0] == i)[0][0]]
            if -i in clustering_hist[0]:
                clustering_neg[i] = clustering_hist[1][np.where(clustering_hist[0] == -i)[0][0]]

        fig, ax = plt.subplots(figsize=(6, 4))
        # matplotlib.rcParams.update({'font.size': 18})
        plt.plot(intonation_pos[1:], color='#66b3ff', linestyle=":", label="Selected clusters: Positive")
        plt.plot(intonation_neg[1:], color='#000099', linestyle="-.", label="Selected clusters: Negative")
        plt.plot(clustering_pos[1:], color='#ff751a', linestyle="--", label='Remaining clusters: Positive')
        plt.plot(clustering_neg[1:], color='#cc2900', linestyle="-", label='Remaining clusters: Negative')
        plt.legend(loc="upper right")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.set_ylabel('Occurrences in 1000s')
        ax.set_xlabel('Deviations (cents)')
        ax.set_xlim(0.5, 100.5)
        ax.set_ylim(0, 250000)
        # for i in range(5, 100, 5):
        #     plt.axvline(x=i, ls='dotted', color="green", linewidth=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir_intonation, "full_pos_vs_neg_line.eps"), format="eps")
        plt.show()

        width = 0.24
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind_pos - width * 1.5, intonation_pos, width, color='#66b3ff')
        rects2 = ax.bar(ind_pos - width * 0.5, intonation_neg, width, color='#000099')
        rects3 = ax.bar(ind_pos + width * 0.5, clustering_pos, width, color='#ff751a')
        rects4 = ax.bar(ind_pos + width * 1.5, clustering_neg, width, color='#cc2900')
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Occurrences in 1000s')
        ax.set_xlabel('Deviations (cents)')
        ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Selected clusters: Positive', 'Selected clusters: Negative',
                'Remaining clusters: Positive', 'Remaining clusters: Negative'))
        plt.ylim(0, 160000)
        plt.xlim(0.5, 100.5)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir_intonation, "full_pos_vs_neg.eps"), format="eps")
        plt.show()


        width = 0.4
        fig, ax = plt.subplots()
        rects3 = ax.bar(ind_pos - width * 0.5, clustering_pos, width, color='#ff751a')
        rects4 = ax.bar(ind_pos + width * 0.5, clustering_neg, width, color='#cc2900')
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Occurrences in 1000s')
        ax.set_xlabel('Deviations (cents)')
        ax.legend((rects3[0], rects4[0]), ('Remaining clusters: Positive', 'Remaining clusters: Negative'))
        plt.ylim(0, 160000)
        plt.xlim(0.5, 60.5)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir_intonation, "clustering_pos_vs_neg.eps"), format="eps")
        plt.show()

        width = 0.4
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind_pos - width * 0.5, intonation_pos, width, color='#66b3ff')
        rects2 = ax.bar(ind_pos + width * 0.5, intonation_neg, width, color='#000099')
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Occurrences in 1000s')
        ax.set_xlabel('Deviations (cents)')
        ax.legend((rects1[0], rects2[0]), ('Selected clusters: Positive', 'Selected clusters: Negative'))
        plt.ylim(0, 160000)
        plt.xlim(0.5, 60.5)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir_intonation, "intonation_pos_vs_neg.eps"), format="eps")
        plt.show()

    if args.bootstrap is True:
        bootstrap(intonation_midi_dir, differences_dir, cents_min=args.bootstrap_cents_min,
                  cents_max=args.bootstrap_cents_max)
        bootstrap(clustering_midi_dir, differences_dir, cents_min=args.bootstrap_cents_min,
                  cents_max=args.bootstrap_cents_max)

    plot_pipeline(differences_dir, plots_dir_intonation)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--get_histogram', default=False, type=utils.str2bool)
    parser.add_argument('--max_count', default=4702, type=int)
    parser.add_argument('--bootstrap', default=False, type=utils.str2bool)
    parser.add_argument('--bootstrap_cents_min', default=1, type=int)
    parser.add_argument('--bootstrap_cents_max', default=20, type=int)
    args = parser.parse_args()
    # create and clear directories
    main(args)
