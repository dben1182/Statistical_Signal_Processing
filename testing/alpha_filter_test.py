#%%

#this file writes up a simple alpha filter and plots the frequency response of it


import scipy
import numpy as np

import matplotlib.pyplot as plt

alpha = np.linspace(0.1, 0.99, num = 10)

#creates the vector of the b coefficients
b = np.array([1.0-alpha[8]])

a = np.array([1.0, -1.0*alpha[8]])

w, h = scipy.signal.freqz(b, a, worN=10000)

plt.plot(w, h)
plt.xscale('log')
plt.yscale('log')

# %%
