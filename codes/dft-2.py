# dft2.py: dft of real signal

import numpy as np
import matplotlib.pyplot as plt

# define the input signal here, compare to k0=7 vs ko=7.5
N = 64
k0 = 7

# Use real signal here by changing from exp (1j) to cos (real) 
# A real signal has two peaks, compare to dft1.py
x = np.cos(2 * np.pi * k0/N * np.arange(N))
# x = np.exp(1j * 2 * np.pi * k0/N * np.arange(N))

## DFT part
# variable for saving frequency components
X = np.array([])
# discrete-time and frequency indexes
nt = np.arange(-N/2, N/2)
nf = np.arange(-N/2, N/2)

# to plot frequency
for k in nf:
    s = np.exp(1j * 2 * np.pi * k/N * nt)
    X = np.append(X, sum(x*np.conjugate(s)))

# plotting, change k0 to see the difference
plt.plot(nf, abs(X))
plt.axis([-N/2, N/2-1, 0, N])    # 0, N-1
plt.show()
