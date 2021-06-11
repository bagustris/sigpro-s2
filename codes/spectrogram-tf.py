#!/usr/bin/env python3.8
# plot stft and spectrogram with tensorflow

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# function to decode/read wav
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# get waveform
def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


# plt.plot(waveform.numpy())
# plt.show()


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    if waveform.numpy().shape[0] < 16000:
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    else:
        zero_padding = tf.zeros([0])
    
  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, 
                                # fft_length=256, 
                                frame_length=256, 
                                frame_step=256)

    spectrogram = tf.abs(spectrogram)
    return spectrogram


# function to plot spectrogram
def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

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
plt.show(block=False)
