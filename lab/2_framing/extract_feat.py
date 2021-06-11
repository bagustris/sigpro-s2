# solution to lab 2

import numpy as np
import librosa
import matplotlib.pyplot as plt

x, fs = librosa.load('../lab/1_sampling/speech.wav', sr=None)

M = XXX         # window size = frame length
N = XXX         # n fft
S = XXX//2      # choose stride/step/hop length to be window size/2

rmse = librosa.feature.XXX(XXX, frame_length=XXX, hop_length=XXX)
zcr = librosa.feature.XXX(x, frame_length=XXX, hop_length=N)
mfcc = librosa.feature.XXX(x, sr=XXX, n_mfcc=13, n_fft=N, hop_length=S)


plt.plot(rmse)
plt.show()