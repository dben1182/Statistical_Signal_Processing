#%%

#this file implements the project 1 again

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

#M represents the number of samples in Theta
M = 50
#N represents the number of random samples to draw
#after a little trial and error, I have found that we need on the order of 10,000
#iterations to make this work.
N = 20


########################################################################################
#section 1
#gets the R matrix, which is toeplitz symmetric


#sets the mean of the MVRN vector
mu = np.zeros((M))

#gets the initial theta vector
theta_initial = np.random.uniform(low=0.0, high=100.0, size=M)

#gets the initial R matrix
R_initial = sp.linalg.toeplitz(theta_initial)

#checks the eigenvalues of R_initial, and if it is negative, we add that amount to the main diagonal

R_initial_eigenvalues, eigenvectors = np.linalg.eig(R_initial)

#gets the minimum eigenvalue of the R_initial_eigenvalues
R_initial_eigenvalues_minimum = np.min(R_initial_eigenvalues)

#prints out the minimum eigenvalue
#print("Initial Minimum Eigenvalue: ", R_initial_eigenvalues_minimum)

#adds the absolute value of that to the real theta plus a buffer of one, with the rest unchanged
theta = theta_initial
theta[0] = theta[0] + np.abs(R_initial_eigenvalues_minimum) + 1.0 

#gets the final R matrix
R = sp.linalg.toeplitz(theta)

#gets the eigenvalues
R_eigenvalues, R_eigenvectors = np.linalg.eig(R)

#gets the minimum eigenvalue of R
R_minimum_eigenvalue = np.min(R_eigenvalues)
#print("Final Minimum Eigenvalue: ", R_minimum_eigenvalue)

#end section 1
########################################################################################

########################################################################################
#section 2
#section 2 gets the weird cursive J matrix, which I really don't understand from this assignment
#gets the fisher information matrix

R_Matrix = np.asmatrix(R)

#to start, we get the flattened R, into a vector M squared in length
R_flattened = np.transpose(R_Matrix.flatten('C'))
#print("R Flattened: \n", R_flattened)

#creates a cursive J Matrix

J_cursive = np.zeros((M*M, M))

#iterates through each element in theta
for i in range(M):
    #iterates through each element in the R_flattened
    for j in range(M*M):
        #if it is the respective value for the flattened R,
        #then it is set to 1.0. else, it stays a zero
        if R_flattened[j] == theta[i]:
            J_cursive[j][i] = 1.0

#print(J_cursive)

#gets the fisher information matrix
#gets the R_inverse
R_inv = np.linalg.inv(R)

J = (M/2.0)*np.transpose(J_cursive) @ np.kron(R_inv, R_inv) @ J_cursive


#gets the CRB matrix
CRB = (1/N)*np.linalg.inv(J)
#end section 2
########################################################################################


########################################################################################
#Begin Section 3
#gets the maximum likelihood estimate of R, and subsequently theta


#creates the R hat maximum likelihood
R_hat = np.zeros((M,M))

#sets the number of Monte Carlo simulations
num_monte_carlo_runs = 10000

#iterates through and gets the R_hat for the N random vector samples
for i in range(num_monte_carlo_runs):

    #gets the X matrix, which is N random vector samples of the distribution
    X = np.transpose(np.random.multivariate_normal(mu, R, size=N))
    #prints the size of X
    #print("X: \n", X)

    #gets the outer product of X with itself
    outer_product = np.outer(X, X)
    #adds each x to itself with 
    R_hat = R_hat + outer_product


#normalizes R_hat
R_hat = (1/N)*R_hat

#gets theta hat from R_hat
theta_hat = R_hat[0,:]

#gets the error, which is theta hat minus theta
theta_error = theta_hat - theta


#end section 3
########################################################################################

########################################################################################
#section 4
#Section 4 includes the plots to compare the actual sample estimation error variance

#gets the Cramer Rao lower bounds for the error variances by extracting the diagonals
CRB_vector = np.diag(CRB)
print("CRB Vector: \n", CRB_vector)

#gets the actual squared error of theta
MSE = (theta_error)**2

#plots the MSE for the thetas
plt.figure()
plt.plot(MSE)
#also plots the CRB on the same plot
plt.plot(CRB_vector)


########################################################################################

# %%
