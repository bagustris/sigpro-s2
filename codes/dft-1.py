# dft1.py: dft of complex signal

import numpy as np
import matplotlib.pyplot as plt


# shape of signal
N = 64      # size of signal, number of sample
k0 = 7      # discrete frekuency index

# define the input signal here, x with exp is complex, x with cos is real
# x = np.exp(1j * 2 * np.pi * k0/N * np.arange(N))
x = np.exp(1j * 2 * np.pi * k0/N * np.arange(N))

# DFT part
X = np.array([])

for k in range(N):  # range(N)
    s = np.exp(1j * 2 * np.pi * k/N * np.arange(N))
    X = np.append(X, sum(x * np.conjugate(s)))

plt.plot(np.arange(N), abs(X))
plt.axis([0, N-1, 0, N])    # 0, N-1
plt.show()


# note:
## complex signal has one peak
## real signal has two peaks, amplitude becomes half
## the x axis is not so intuitive now, should be 7 and -7
