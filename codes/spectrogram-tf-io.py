#!/usr/bin/env python3.8
# plot stft and spectrogram with tensorflow

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# lay read audio file
audio = tfio.audio.AudioIOTensor('../lab/1_sampling/speech.wav')

# read as tensor
audio_tensor = tf.squeeze(audio.to_tensor(), axis=[-1])

tensor = tf.cast(audio_tensor, tf.float32) / 32768.0

# calculate spectrogram
spectrogram = tfio.audio.spectrogram(
    tensor, nfft=512, window=512, stride=256)

# Convert to mel-spectrogram
mel_spectrogram = tfio.audio.melscale(
    spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

# Convert to db scale mel-spectrogram
dbscale_mel_spectrogram = tfio.audio.dbscale(
    mel_spectrogram, top_db=80)

plt.figure()
plt.plot(tensor.numpy())
plt.figure()
plt.imshow(tf.math.log(spectrogram).numpy().T, origin='lower')
plt.figure()
plt.imshow(dbscale_mel_spectrogram.numpy().T, origin='lower')
plt.figure()
plt.imshow(tf.math.log(mel_spectrogram).numpy().T, origin='lower')

plt.show()