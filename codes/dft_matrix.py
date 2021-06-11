# python code to demonstrate DFT, DFT matrix, and IDFT
# this script is intended to be run per line

import numpy as np

# create matrix
a = np.eye(4)

# compute fft/dft
s4 = np.fft.fft(a)

# Compute conjugate s4*
s4_conj = np.conj(s4)   # or s4.conj()
print(s4_conj)

# show that s4* = IDFT
s4_idft = s4_conj @ s4
print(s4_idft)

## another number
b = np.array([1, 2, 3, 4])
print(b)

## with fft
b_dft = np.fft.fft(b)
print(b_dft)

## with dft matrix
b_dftm = s4 @ b
print(b_dftm)

## check if same
print(b_dft == b_dftm)

## another number in slide
c = np.array([1, -1, 1, -1])

## calculate fft/dft and idft
c_dft = np.fft.fft(c)
print(c_dft)

c_idft = np.fft.ifft(c)
print(c_idft)

## idft direct computation
s4_ifft = np.fft.ifft(a)
print(s4_ifft)
