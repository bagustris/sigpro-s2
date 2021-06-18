# plot spectrogram using numpy
# source: https://courses.engr.illinois.edu/ece590sip/sp2018/spectrograms1_wideband_narrowband.html


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np   # We use this one to do numerical operations


# function to slices a single audio file into frames
def enframe(x, S, L):
    w = np.hamming(L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0, nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return frames

# function to calculate stft: dft in all frames
def stft(frames, N, Fs):
    stft_frames = [np.fft.fft(x, N) for x in frames]
    freq_axis = np.linspace(0, Fs, N)
    return (stft_frames, freq_axis)

# function to convert to power
def stft2level(stft_spectra, max_freq_bin):
    magnitude_spectra = [abs(x) for x in stft_spectra]
    max_magnitude = max([max(x) for x in magnitude_spectra])
    min_magnitude = max_magnitude / 1000.0
    for t in range(0, len(magnitude_spectra)):
        for k in range(0, len(magnitude_spectra[t])):
            magnitude_spectra[t][k] /= min_magnitude
            if magnitude_spectra[t][k] < 1:
                magnitude_spectra[t][k] = 1
    level_spectra = [20*np.log10(x[0:max_freq_bin]) for x in magnitude_spectra]
    return level_spectra


# read audio file
fs, x = wavfile.read('../lab/1_sampling/speech.wav')

# define window parameters
frame_length = 512                  # Change this for narrow/wideband
fft_length = 512
frame_skip = frame_length//2        # hop length
max_freq = 5000

# calculate each step: enframe, stft, stft2level, and spectrogram
frames = enframe(x, frame_skip, frame_length)
(spectra, freq_axis) = stft(frames, fft_length, fs)
sgram = stft2level(spectra, int(max_freq*fft_length/fs))
max_time = len(frames)*frame_skip/fs

plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(sgram)), 
    origin='lower', extent=(0,max_time, 0,max_freq),
    aspect='auto')

plt.xlabel('Time (ms)')
plt.ylabel('Freq (Hz)')
plt.show()