# Sleep Stage Classification with Deep Learning

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) [![PythonVersion](https://camo.githubusercontent.com/fcb8bcdc6921dd3533a1ed259cebefdacbc27f2148eab6af024f6d6458d5ec1f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e36253230253743253230332e37253230253743253230332e38253230253743253230332e392d626c7565)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)

## Introduction

We propose to use Dreem 2 headband data to perform sleep stage scoring on 30 seconds epochs of biophysiological signals using deep learning techniques.
This project was done in the context of the Deep Learning course from the MVA master and CentraleSupelec, the final report is available [here](final_report_DL_JQ.pdf).


## Data

The data can be downloaded [here](https://www.kaggle.com/c/dreem-3-sleep-staging-challenge-2021/data), and must be unzip and moved to `.\Data\raw_data`

The data is separated between the differents signals recorded by the Dreem headband. Signal is shapes in windows of 30 seconds: - train dataset has 15000 windows
- test dataset has 6000 windows

Electroencephalogram is measured at 5 differents locations of the head (eeg_1, eeg_2, eeg_4,... eeg_6). The sampling frequency is 50 Hz.

Accelerometer ([x/y/z]) channels are sampled at 10Hz

eeg_1 - EEG (derivation F7-O1) sampled at 50 Hz -> 1500 values
eeg_2 - EEG (derivation F8-O2) sampled at 50 Hz -> 1500 values
eeg_4 - EEG (derivation F8-F7) sampled at 50 Hz -> 1500 values
eeg_5 - EEG (derivation F8-O1) sampled at 50 Hz -> 1500 values
eeg_6 - EEG (derivation F7-O2) position sampled at 50 Hz -> 1500 values
x - Accelerometer along x axis sampled at 10 Hz -> 300 values
y - Accelerometer along y axis sampled at 10 Hz -> 300 values
z - Accelerometer along z axis sampled at 10 Hz -> 300 values


## Structure

```

│   .gitignore
│   README.md
│   requirements.txt
│   run_models.ipynb
│   
├───Data
│   ├───pre_processed_data
│   │       Multitaper_eeg_test.npy
│   │       Multitaper_eeg_train.npy
│   │       Multitaper_position_train.npy
│   │       Multitaper_position_test.npy
│   │   
│   └───raw_data
│       │   sample_submission
│       │   X_test.h5
│       │   X_train.h5
│       │   y_train.csv
│       │   
│   
│   
└───src
    │   setup.py
    │   
    ├───sleep_classif
    │   │   CNNadvanced.py
    │   │   CNNmodel.py
    │   │   CNNmultitaper.py
    │   │   dataloaders.py
    │   │   LSTMConv.py
    │   │   preprocessing.py
    │   │   trainer.py
    

   
```

## Run code

Run the different models and data processings by running `run_models.ipynb`


To run the attnsleep model please refer to https://github.com/emadeldeen24/AttnSleep (you need to convert the eeg signals files to npz files before running the model)
