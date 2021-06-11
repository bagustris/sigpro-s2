# stft demo with librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load('../lab/1_sampling/speech.wav', sr=None)
fig, ax = plt.subplots()

# define size of window and FFT
M = 32          # window size, smaller:more resolution
N = 512         # fft number

# calculate STFT
S = np.abs(librosa.stft(y, n_fft=N, win_length=M,
                        hop_length=int(M/2)))

img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr,
                               hop_length=int(M/2), y_axis='log', fmax=5000,
                               fmin=300, x_axis='time', ax=ax)

ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.show()
