# idft.py: compute inverse dft

import numpy as np
import matplotlib.pyplot as plt

N = 64
k0 = 7

# define the signal here, change from exp (1j) to cos (real) 
x = np.cos(2 * np.pi * k0/N * np.arange(N))
X = np.array([])
nf = np.arange(-N/2, N/2)

for k in nf:  # range(N)
    s = np.exp(1j * 2 * np.pi * k/N * nf)    # 
    X = np.append(X, sum(x * np.conjugate(s))) # spectrum

## IDFT part
y = np. array([])
for n in nf:
    s = np.exp(1j * 2 * np.pi * n/N * nf)
    y = np.append(y, 1.0/N * sum(X * s))

plt.plot(nf, y)
plt.axis([-N/2, N/2-1, -1, 1])
plt.show()
