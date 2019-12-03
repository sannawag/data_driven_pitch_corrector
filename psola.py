"""
Author: Sanna Wager
Created on: 9/18/19

https://github.com/sannawag/TD-PSOLA

References:
- https://www.surina.net/article/time-and-pitch-scaling.html, and corresponding library https://gitlab.com/soundtouch
- https://courses.engr.illinois.edu/ece420/lab5/lab/#overlap-add-algorithm
- Charpentier, Ff, and M. Stella. "Diphone synthesis using an overlap-add
  technique for speech waveforms concatenation."
  ICASSP'86. IEEE International Conference on Acoustics, Speech,
  and Signal Processing. Vol. 11. IEEE, 1986.

The signal is expected to be voiced.
Peak detection may cause bad results for unvoiced audio.
"""

from globals import hopSize, global_fs
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import librosa

plt.style.use('ggplot')


def psola_shift_pitch(signal, fs, f_ratio_list):
    """
    Calls psola pitch shifting algorithm
    :param signal: original signal in the time-domain
    :param fs: sample rate
    :param f_ratio: list of ratios by which the frequency will be shifted
    :return: list of pitch-shifted signals
    """
    peaks = find_peaks(signal, fs, max_change=1.005, min_change=0.995)

    new_signal_list = []
    for f_ratio in f_ratio_list:
        new_signal_list.append(psola(signal, peaks, f_ratio))
    return new_signal_list


def find_peaks(signal, fs, max_hz=950, min_hz=75, analysis_win_ms=40, max_change=1.5, min_change=0.6):
    """
    Find sample indices of peaks in time-domain signal
    :param max_hz: maximum measured fundamental frequency
    :param min_hz: minimum measured fundamental frequency
    :param analysis_win_ms: window size used for autocorrelation analysis
    :param max_change: restrict periodicity to not increase by more than this ratio from the mean
    :param min_change: restrict periodicity to not decrease by more than this ratio from the mean
    :return: peak indices
    """
    N = len(signal)
    min_period = fs // max_hz
    max_period = fs // min_hz
    sequence = int(analysis_win_ms / 1000 * fs)  # analysis sequence length in samples
    periods = compute_periods_per_sequence(signal, sequence, min_period, max_period)
    # simple hack to avoid glitches: assume that the pitch should not vary much, restrict
    mean_period = np.median(periods)
    max_period = int(mean_period * 1.1)
    min_period = int(mean_period * 0.9)
    periods = compute_periods_per_sequence(signal, sequence, min_period, max_period)
    # find the peaks
    peaks = [np.argmax(signal[:int(periods[0]*1.1)])]
    while True:
        prev = peaks[-1]
        idx = min(prev // sequence, len(periods) - 1)  # current autocorrelation analysis window
        if prev + int(periods[idx] * max_change) >= N:
            break
        # find maximum near expected location
        peaks.append(prev + int(periods[idx] * min_change) +
                np.argmax(signal[prev + int(periods[idx] * min_change): prev + int(periods[idx] * max_change)]))
    return np.array(peaks)


def compute_periods_per_sequence(signal, sequence, min_period, max_period):
    """
    Computes periodicity of a time-domain signal using autocorrelation
    :param sequence: analysis window length in samples. Computes one periodicity value per window
    :param min_period: smallest allowed periodicity
    :param max_period: largest allowed periodicity
    :return: list of measured periods in windows across the signal
    """
    offset = 0  # current sample offset
    periods = []  # period length of each analysis sequence
    N = len(signal)
    while offset < N - max_period:
        fourier = fft(signal[offset: offset + sequence])
        fourier[0] *= 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        autoc_peak = min_period + np.argmax(autoc[min_period: max_period])
        periods.append(autoc_peak)
        offset += sequence
    return periods


def psola(signal, peaks, f_ratio):
    """
    Time-Domain Pitch Synchronous Overlap and Add
    :param signal: original time-domain signal
    :param peaks: time-domain signal peak indices
    :param f_ratio: pitch shift ratio
    :return: pitch-shifted signal
    """
    N = len(signal)
    # Interpolate
    new_signal = np.zeros(N)
    new_peaks_ref = np.linspace(0, len(peaks) - 1, len(peaks) * f_ratio)
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)
    for i in range(len(new_peaks)):
        weight = new_peaks_ref[i] % 1  # keep only decimals
        left = np.floor(new_peaks_ref[i]).astype(int)  # left peak index
        right = np.ceil(new_peaks_ref[i]).astype(int)  # right peak index
        new_peaks[i] = int(peaks[left] * (1 - weight) + peaks[right] * weight)
    # PSOLA
    for j in range(len(new_peaks)):
        # find the corresponding old peak index
        i = np.argmin(np.abs(peaks - new_peaks[j]))
        # get the distances to adjacent peaks
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j-1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j+1] - new_peaks[j]]
        # edge case truncation
        if peaks[i] - P1[0] < 0:
            P1[0] = peaks[i]
        if peaks[i] + P1[1] > N - 1:
            P1[1] = N - 1 - peaks[i]
        # linear OLA window
        window = list(np.linspace(0, 1, P1[0] + 1)[1:]) + list(np.linspace(1, 0, P1[1] + 1)[1:])
        # center window from original signal at the new peak
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]
    return new_signal


if __name__=="__main__":
    # Load audio
    orig_signal, fs = librosa.load("female_scale.wav", sr=44100)
    N = len(orig_signal)
    # Pitch shift amount as a ratio
    f_ratio = 2 ** (0.07 / 12)
    # Shift pitch
    new_signal = psola_shift_pitch(orig_signal, fs, f_ratio)
    # Plot and write to disk
    plt.plot(orig_signal[:-1])
    plt.show()
    plt.plot(new_signal[:-1])
    plt.show()
    librosa.output.write_wav("female_scale_transposed_{}.wav".format(f_ratio), new_signal, fs)
