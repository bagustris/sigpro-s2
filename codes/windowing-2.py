import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

M = 64              # lebar window = 64
N = 1024            # fft size = 1024
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1, figsize=(9.5, 6))
# window, ganti dengan ones, hanning, bartlett, blackman, hamming
fftbuffer[hN-hM:hN+hM] = np.hamming(M)   
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b', lw=1.5)
plt.axis([-hN, hN, 0, 1.1])
plt.xlabel('waktu (detik)')
plt.title('windowing M = {}'.format(M))


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]


plt.subplot(2,1,2)
plt.plot(np.linspace(-hM, hM, len(mX)), mX1-max(mX), 'r', lw=1.5)
plt.axis([-hM, hM, -100, 0])
plt.xlim([-20, 20])
plt.title('DFT dari window N = {}'.format(N))
plt.xlabel('Frekuensi bins')

plt.tight_layout()
plt.show()
