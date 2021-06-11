# plot bagian real dari sinyal sinusoid kompleks
import matplotlib.pyplot as plt
import numpy as np

N = 500     # N DFT, change to view effect
k = 3       # frekuensi
n = np.arange(-N/2, N/2)                # time index
s = np.exp(1j * 2 * np.pi * k * n / N)  # complex sinusoid 

plt.plot(n, np.real(s))
plt.axis([-N/2, N/2, -1, 1])
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()
