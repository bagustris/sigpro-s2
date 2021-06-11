# plot sinyal sinusoid di Python 
import matplotlib.pyplot as plt
import numpy as np

A = 0.8         # amplutido
f0 = 1000       # frekuensi
phi = np.pi/2   # fasa
fs = 44100      # sampling frekuensi
t = np.arange( -0.002, 0.002, 1.0/fs)   # time series
x = A * np.cos (2 * np.pi * f0 * t * phi)   # sinyal x

plt.plot(t, x)
plt.axis([-0.002, 0.002, -0.8, 0.8])
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()
