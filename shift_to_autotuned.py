
import csv
import librosa
import numpy as np
from globals import *
import matplotlib.pyplot as plt
from psola import psola_shift_pitch
from interpolate_pyin import sec_to_frame


def compute_shift_amount_to_equal_tempered(hz):
    midi_index = np.round(12 * np.log2(hz / 440) + 69)
    target = 440 * np.power(2, (midi_index - 69) / 12)
    shift = target - hz
    ratio = target / hz

    return shift, ratio


def apply_autotune_corrections(pyin, pyin_notes, audio, autotune_shifts, random_shift=False):
    # compute shift amount one note at a time
    frames = len(pyin)
    note_beginnings = np.array([note[0] for note in pyin_notes])
    note_endings = np.array([note[1] for note in pyin_notes])
    pitches = np.array([note[2] for note in pyin_notes])

    # compute equal-tempered output for comparison
    frame_shifts_et = np.zeros(frames)
    frame_ratios_et = np.zeros(frames) - 1  # store silence as negative values
    note_shifts_et = np.zeros(len(note_beginnings))
    ratios_et = np.zeros(len(note_beginnings))
    for i in range(len(note_beginnings)):
        mean_hz = pitches[i]
        note_shifts_et[i], ratios_et[i] = compute_shift_amount_to_equal_tempered(mean_hz)
        frame_shifts_et[note_beginnings[i]: note_endings[i]] = note_shifts_et[i]
        frame_ratios_et[note_beginnings[i]: note_endings[i]] = ratios_et[i]

    note_shifts = np.zeros(len(note_beginnings))
    ratios = np.zeros(len(note_beginnings))
    for i in range(len(note_beginnings)):
        mean_hz = pitches[i]
        if random_shift:
            ratios[i] = 2 ** ((np.random.random(1)[0] * 2 - 1) * 100 / 1200)
        else:
            ratios[i] = 2 ** (autotune_shifts[i] * 100 / 1200)  # convert from semitones to ratios
        shifted_pitch = mean_hz * ratios[i]
        note_shifts[i] = mean_hz - shifted_pitch

    plt.plot(note_shifts_et[:20], 'o', color="blue")
    plt.plot(note_shifts[:20], 'o', color="green")
    plt.show()
    frame_shifts = np.zeros(frames)
    frame_ratios = np.zeros(frames) - 1  # store silence as negative values
    for i in range(len(note_beginnings)):
        frame_shifts[note_beginnings[i]: note_endings[i]] = note_shifts[i]
        frame_ratios[note_beginnings[i]: note_endings[i]] = ratios[i]

    # examine the resulting pitch contour
    shifted_pyin = np.copy(pyin) + frame_shifts

    # apply the shift to the audio
    shifted_audio = np.copy(audio)
    # shifted_audio = np.zeros(len(audio))
    margin = 3
    for i in range(len(note_beginnings)):
        if i == 0:
            start_sample = max((note_beginnings[i] - margin) * hopSize, 0)
        else:
            start_sample = max((note_beginnings[i] - margin) * hopSize, note_endings[i - 1] * hopSize)
        if i == len(note_beginnings) - 1:
            end_sample = min((note_endings[i] + margin) * hopSize, len(audio))
        else:
            end_sample = min((note_endings[i] + margin) * hopSize, note_beginnings[i + 1] * hopSize)
        temp = psola_shift_pitch(
            audio[start_sample: end_sample], fs=global_fs, f_ratio_list=[ratios[i]])[0]
        temp[0: margin * hopSize] *= np.linspace(0, 1, margin * hopSize)
        temp[-margin * hopSize: -1] *= np.linspace(1, 0, margin * hopSize - 1)
        shifted_audio[start_sample: start_sample + margin * hopSize] *= np.linspace(1, 0, margin * hopSize)
        shifted_audio[end_sample - margin * hopSize: end_sample] *= np.linspace(0, 1, margin * hopSize)
        shifted_audio[start_sample + margin * hopSize: end_sample - margin * hopSize] = 0
        shifted_audio[start_sample: end_sample] += temp
    return shifted_pyin, shifted_audio, frame_shifts, frame_ratios


def parse_note_csv(fpath, convert=True):
    with open(fpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        notes = []
        for row in csv_reader:
            start = np.float32(row[0])
            duration = np.float32(row[2])
            pitch = np.float32(row[1])
            if convert:
                notes.append([sec_to_frame(start), sec_to_frame(start + duration), pitch])
            else:
                notes.append([start, start + duration, pitch])
    return notes


def main():
    extension = "3"
    autotune_results_dir = "./results_root_" + + extension + "/deep_autotuner_outputs"

    pyin_csv_dir = "./Intonation/pyin"
    for fname in os.listdir(pyin_csv_dir):
        if not fname.endswith(".npy"):
            continue
        song_id = fname.replace("_vocals.npy", "")
        root_path = "./"
        autotune_result_path = os.path.join(root_path, "deep_autotuner_outputs/results_subjective_test_{}/"
                "testing_epoch_0_outputs_{}_vocals.npy".format(extension, song_id))
        back_path = os.path.join(root_path, "backing_tracks_wav/{}_back.wav".format(song_id))
        wav_path = os.path.join(root_path, "vocals/{}_vocals.wav".format(song_id))
        pyin_path = os.path.join(root_path, "pyin/{}_vocals.npy".format(song_id))
        note_pyin_path = os.path.join(root_path, "notes_pyin/{}_vocals.csv".format(song_id))
        output_wav_path = os.path.join(root_path, "{}_mix_parsed/{}_mix_quiz.wav".format(extension, song_id))
        pyin = np.load(pyin_path)
        pyin_notes = parse_note_csv(note_pyin_path)
        autotune_result = np.load(autotune_result_path)[:, 0] * -1  # keep first version of 7, invert the direction
        audio, _ = librosa.load(wav_path, sr=global_fs)
        shifted_pyin, shifted_audio, frame_shifts, frame_ratios = apply_autotune_corrections(
            pyin, pyin_notes, audio, autotune_result, random_shift=False)  # set to true to create quiz example
        back, _ = librosa.load(back_path, sr=global_fs)
        length = min(len(shifted_audio), len(back), len(audio))
        librosa.output.write_wav(output_wav_path, shifted_audio[:length] + back[:length], sr=global_fs)


if __name__=="__main__":
    main()
