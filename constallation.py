#%%

import numpy as np


#this file calculates the energy of certain symbols for a binary communicaiton protocol


#defines the parameters of the function


A = 10
k = 5
N = 100

#defines the variance

variance = 1.5

#creates the covariance matrix

v = variance*np.ones(N)

R = np.diag(v)


#creates the two vectors for the base symbol

s = np.zeros([N, 1])

for i in range(N):
    s[i][0] = A*np.cos(2*np.pi*(k/N)*i)



#sets the m0 and m1 vectors
    
m0 = ((-1.0)**0)*s

m1 = ((-1.0)**1)*s


#gets the energy



Es = np.abs(np.transpose(m0) @ np.linalg.inv(R) @ m0)
print(Es)

# %%
