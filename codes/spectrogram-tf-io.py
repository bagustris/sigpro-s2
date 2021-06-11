#!/usr/bin/env python3.8
# plot stft and spectrogram with tensorflow

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np



# main program to plot
file_path = '../lab/1_sampling/speech.wav'
waveform = get_waveform(file_path)
spectrogram = get_spectrogram(waveform)
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, len(waveform.numpy())])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
# axes[1].set_xlim([0, len(waveform.numpy())])
plt.show()
