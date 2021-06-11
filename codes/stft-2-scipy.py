# stft demo with librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from  scipy import signal
from  scipy.io import wavfile


sr, x = wavfile.read('../lab/1_sampling/speech.wav')

# define size of window and FFT
M = 256
N = 512

# calculate spectrogram
sf, st, sd = signal.spectrogram(x, sr, window='hann', nperseg=M, noverlap=M/2,
                                scaling='spectrum', nfft=N)  

plt.pcolormesh(st, sf, sd)
plt.ylabel('Frequency [Hz]')
plt.ylim([0, 2000])
plt.xlabel('Time [sec]')

plt.show()
