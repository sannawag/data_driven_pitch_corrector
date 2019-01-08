# Overview

This program outputs note-wise constant pitch corrections of up to 100 cents (one semitone).
It takes as input vocals and backing tracks in separate wav files along with a
frame-wise pitch analysis of the vocals. The corrections are 
applied along a continuous scale and are trained here based on examples of in-tune singing. The program
pre-processes the data by detecting the note onsets and offsets in the vocals and randomly shifting 
every note while storing the shift in cents as ground truth for the program output. 

The paper related to this program program, along with audio examples, is available [here](http://homes.sice.indiana.edu/scwager/deepautotuner.html).

# Dataset

The program was trained using the Intonation dataset. More information on the full dataset used
for the paper and how to access it via the Stanford DAMP can be found [here](http://homes.sice.indiana.edu/scwager/images/damp_dataset_nov5.pdf).

Other data with the same format can also be used to populate the directories in Intonation/. The data needs to have wav files of the separate vocals and backing tracks and a frame-wise pitch tracking analysis of the vocals. pYIN pitch analysis has the high
resolution required for this program, where corrections are in cents. The frame and hop lengths for all
analyses must be the same and is defined in globals.py. The file intonation.csv requires a list of the file names of the vocal and backing track file names without the ".wav" extension. Variable boundary_arr_id in rnn.py should be changed accordingly. This variable determines the split between training and validation data. 

More information on pitch tracking can be found at:

M. Mauch and S. Dixon, “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions,”
in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2014), 2014.

# Running the program

The main program is rnn.py. It can be run as follows:
`python rnn.py --learning_rate 0.000005 --resume False --num_layers 1 --small_dataset True`

Three pre-trained models are available in the pytorch_checkpoints_and_models directory. Please check the
README in the directory for more details on how to import them into the program. 

Directories for input and output are defined in globals.py. Most of them are created automatically when
running rnn.py and reset on subsequent runs.

synthesize.py can be used to hear the results of the program.

preprocessed_data_plotter.py and the programs in results_plotting_programs are used to plot the input
data and results (e.g., predictions, losses). When running rnn.py, some plots are automatically generated
and stored in results_root/plots. Results of note parsings (detecting the onset and offset of every note)
are in plots/.

# Reference 

If you use this code, please use the following reference:

S. Wager, G. Tzanetakis, C. Wang, L. Guo, A. Sivaraman, and M. Kim,
"Deep Autotuner: A data-driven approach to natural sounding pitch correction for singing voice
in karaoke performances,"
in IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Submitted for publication.

The reference to the dataset creation paper is:

S. Wager, G. Tzanetakis, C. Wang, S. Sullivan, J. Shimmin, M. Kim, and P. Cook, 
"Intonation: A dataset of quality vocal performances refined by spectral clustering on pitch congruence,"
in IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Submitted for publication.

