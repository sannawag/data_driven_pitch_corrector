
"""Counts average number of notes per performance in dataset"""

import numpy as np
import os
import pickle

dir = "/share/project/scwager/autotune_fa18_data/Intonation/autotune_preprocessed"

note_count = 0
counter = 0
for f in os.listdir(dir):
    if not f.endswith("pkl"):
        continue
    try:
        with open(os.path.join(dir, f), "rb") as file:
            data_dict = pickle.load(file)
            note_count += len(data_dict['shifts_gt'][0, :])
            counter += 1
            avg = note_count / counter
            print(counter, ": average = ", avg)
    except Exception as e:
        continue
