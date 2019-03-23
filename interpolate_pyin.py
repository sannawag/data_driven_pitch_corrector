
import csv
import numpy as np
from globals import *
import matplotlib.pyplot as plt

"""
Processes output of pYIN pitch analysis: for example, the output of the Sonic Visualizer program with pYIN vamp
plugin. The output of the plugin consists of unevenly spaced timestamps. This program requires the path to the output
file in CSV format. It will generates a frame-by-frame pitch analysis with equal time intervals as a numpy binary
file. This program's output is used in rnn.py.
"""

pyin_csv_path = "/path/to/my_input_file.csv"
output_npy_path = "/path/to/my_output_file.npy"


def sec_to_frame(seconds):
  '''converts time in seconds to time in frames with window and hop size defined above'''
  samples = seconds * global_fs
  frame_idx = (samples // hopSize).astype(int)  # floor division: choose beginning of the frame
  return frame_idx


def main():
    time = np.array([])
    freq = np.array([])
    with open(pyin_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            t = np.float32(row[0])
            f = np.float32(row[1])
            time = np.hstack((time, t))
            freq = np.hstack((freq, f))
            line_count += 1
    plt.plot(time, freq, '.')
    plt.show()

    freq_frames = np.zeros(sec_to_frame(np.max(time)) + 1)
    for i, s in enumerate(time):
        f = sec_to_frame(s)
        freq_frames[f] = freq[i]
    plt.plot(freq_frames)
    plt.show()

    for i in range(1, len(freq_frames) - 1):
        if freq_frames[i-1] > 0 and freq_frames[i+1] > 0:
            freq_frames[i] = (freq_frames[i-1] + freq_frames[i+1]) / 2.0

    plt.plot(freq_frames[:2000])
    plt.show()

    start_sec = np.array([30])
    start_idx = sec_to_frame(start_sec)
    end_sec = np.array([90])
    end_idx = sec_to_frame(end_sec)
    freq_frames = freq_frames[start_idx[0]:end_idx[0]]
    plt.plot(freq_frames)
    plt.show()

    print(freq_frames)
    np.save(output_npy_path, freq_frames)
    return


if __name__ == "__main__":
    main()
