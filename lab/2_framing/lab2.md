The goal of this lab is following:

signal --> windowing --> framing extract features 

1. feature to extract per frame:  
- energy
- zero-crossing rate
- mfcc (13 coefficients)

2. plot each feature
3. concatenate all feature to be single matrix
feat = [energy, zcr, mfcc]  # total 15 feature/columns
