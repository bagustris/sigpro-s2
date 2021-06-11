# dft-5: add unwrap signal
import  numpy as np
from scipy.signal import triang
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.io import wavfile


M = 501
hM1 = int(np.floor(M+1)/2)
hM2 = int(np.floor(M/2))

[fs, x] = wavfile.read('../lab/1_sampling/speech.wav')

x1 = x[5000:5000+M]*np.hamming(M)

N = 1024
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = x1[hM2:]
fftbuffer[N-hM2:] = x1[:hM2]

X = fft(fftbuffer)

# magnitude of real signals are symetry
mX = 20*np.log10(abs(X))

# unwrap phase spectum
pX = np.unwrap(np.angle(X))

plt.subplot(411); plt.plot(x)
plt.subplot(412); plt.plot(X)
plt.subplot(413); plt.plot(mX[:512])
plt.subplot(414); plt.plot(pX)
plt.show()
