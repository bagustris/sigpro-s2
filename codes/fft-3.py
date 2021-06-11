import  numpy as np
from scipy.signal import triang
from scipy.fftpack import fft
import matplotlib.pyplot as plt

x = triang(15)

# center around zero = inverse triangle 
fftbuffer = np.zeros(15)
fftbuffer[:8] = x[7:]
fftbuffer[8:] = x[:7]

X = fft(fftbuffer)

# magnitude of real signals are symetry
mX = abs(X)

# phase spectum, zero phase 
pX = np.angle(X)

plt.subplot(311); plt.plot(fftbuffer)
plt.subplot(312); plt.plot(X)
plt.subplot(313); plt.plot(pX)
plt.show()
