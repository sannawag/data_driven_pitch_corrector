
# Description

This program computes automatic pitch correction for vocal performances. It outputs note-wise constant
pitch shift values up to 100 cents, equivalent to one semitone. It it can also apply the shifts to
the audio. 

The program is trained on examples of in-tune singing and applies corrections along a continuous frequency scale.  

A pre-trained model is available.

# Usage

## Requirements

Requirements are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`. 
Data pre-processing requires a program that computes probabilistic YIN (pYIN) pitch analysis. One option is 
[Sonic Annotator](https://code.soundsoftware.ac.uk/projects/sonic-annotator/wiki). 

## Running the program

To run the program on an example included in the repo using a pre-trained model, run:

`python rnn.py --extension "3" --resume True --run_training False --run_autotune True`

This will output results to directory `results_root_3/realworld_audio_output`, which will contain the original 
performance
(`test_mix.wav`), and the output of the program (`corrected_mix.wav`). Note that the backing track is not 
publicly available, so the backing track wav file is silent (a 10-hz sine wave). For this particular example, 
the program instead loads the pre-computed constant-Q transform (CQT). 
Make sure to first download and uncompress the CQT zip file available at
[http://homes.sice.indiana.edu/scwager/images/survive_4_back_cqt.npy.zip](http://homes.sice.indiana.edu/scwager/images/survive_4_back_cqt.npy.zip)
and place it in `./Intonation/realworld_data/raw_audio/backing_tracks_wav`.

More generally, the program can be run either for training, testing, or auto-tuning by setting
the boolean args `--run_training`, `--run_training`, and `--run_autotune`.
In the case of training and testing,
the dataloader takes as input in-tune singing, detunes the notes of the vocal track, and learns to predict the 
de-tuning amount. In the case
of autotuning, the program takes as input real-world performances, predicts corrections for them, and
synthesizes the output.

Multiple other settings and parameters are available in the argument parser in `rnn.py`.

Define input and output directories along with data format settings in `globals.py`

A pre-trained model is available in the `pytorch_checkpoints_and_models` directory. The `README` in the 
directory provides more details about it.

The program can be run on CPU but runs faster on GPU. 

## Data pre-processing

The program requires frame- and note-wise pYIN pitch analyses. Please check directory 
`./Intonation/realworld_data/pyin` for examples of these. The outputs of the Sonic Visualizer are converted from 
seconds to frame indices.

## Dataset

The `Intonation` directory needs to contain wav files of the vocals and backing 
tracks. The data format should be defined in `globals.py`. `intonation.csv` should contain a list of
the vocal files and corresponding backing track file names (removing the `.wav` extension). Set variables 
`boundary_id_val` and `boundary_id_test`
in `rnn.py` to determine the split between training, testing, and validation data.

The program was trained using the Intonation dataset. More information on the full dataset used
for the paper and how to access it via the Stanford DAMP can be 
found [here](http://homes.sice.indiana.edu/scwager/images/damp_dataset_nov5.pdf). Note that the dimension of the
backing track CQT in the dataset is different from the one the current program is set to.

Any other data can be used instead. 

# References 

If you use this code, please refer to the following:

S. Wager, G. Tzanetakis, C. Wang, and M. Kim,
"Deep Autotuner: A pitch correcting network for singing performances,"
in IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Submitted for publication.

The reference to the dataset creation paper is:

S. Wager, G. Tzanetakis, C. Wang, S. Sullivan, J. Shimmin, M. Kim, and P. Cook, 
"Intonation: A dataset of quality vocal performances refined by spectral clustering on pitch congruence,"
in IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Submitted for publication.

More information on pitch tracking can be found at:

M. Mauch and S. Dixon, “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions,”
in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2014), 2014.

