import  numpy as np
from scipy.signal import triang
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# plot triangle
x = triang(15)
X = fft(x)

# magnitude of real signals are symetry
mX = abs(X)

# phase spectum
pX = np.angle(X)

plt.subplot(411); plt.plot(x)
plt.subplot(412); plt.plot(X)
plt.subplot(413); plt.plot(mX)
plt.subplot(414); plt.plot(pX)
plt.show()
