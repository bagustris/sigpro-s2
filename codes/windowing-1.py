import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# Spek dari window
# Lebih mudah untuk memakai scipy.namawindow, misal: signal.hann

M = 64                              # size of window
window = get_window('hann', M)
hM1 = int(np.floor(M+1)/2)
hM2 = int(np.floor(M/2))

N = 512                             # size of FFT
hN = int((N+1)/2)                   # originally N/2
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = window[hM2:]
fftbuffer[N-hM2:] = window[:hM2]

X = fft(fftbuffer)
absX = abs(X)
absX[absX<np.finfo(float).eps] = np.finfo(float).eps
mX = 20 * np.log10(absX) 
pX = np.angle(X)

mX1 = np.zeros(N)
pX1 = np.zeros(N)
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]
pX1[:hN] = pX[hN:]
pX1[N-hN:] = pX[:hN]

plt.plot(np.arange(-hN, hN)/float(N)*M, mX1-max(mX1))
plt.axis([-20, 20, -100, 0])
plt.show()

