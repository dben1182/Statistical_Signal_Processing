#%%

#this file implements the project 1 again

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

#gets the maximum likelihood estimator
from estimators import ml_estimator
#gets the ad hoc estimator
from estimators import ad_hoc_estimator

#M represents the number of samples in Theta
M = 6
#N represents the number of random samples to draw
#after a little trial and error, I have found that we need on the order of 10,000
#iterations to make this work.
N = 10000


########################################################################################
#section 1
#gets the R matrix, which is toeplitz symmetric


#sets the mean of the MVRN vector
mu = np.zeros((M))

#gets the initial theta vector
theta_initial = np.random.normal(1.0, 1.0, size=M)

#gets the initial R matrix
R_initial = sp.linalg.toeplitz(theta_initial)
print("R initial: \n", R_initial)

#checks the eigenvalues of R_initial, and if it is negative, we add that amount to the main diagonal

R_initial_eigenvalues, eigenvectors = np.linalg.eig(R_initial)

#gets the minimum eigenvalue of the R_initial_eigenvalues
R_initial_eigenvalues_minimum = np.min(R_initial_eigenvalues)

#prints out the minimum eigenvalue
#print("Initial Minimum Eigenvalue: ", R_initial_eigenvalues_minimum)

#adds the absolute value of that to the real theta plus a buffer of one, with the rest unchanged
theta = theta_initial
theta[0] = theta[0] + np.abs(R_initial_eigenvalues_minimum) + 3.0

#gets the final R matrix
R = sp.linalg.toeplitz(theta)


values, vectors = np.linalg.eig(R)
print("Eigenvalues: ", values)

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


#creates the C estimate matrix, which is the covariance of theta error
C = np.zeros((M,M))

#creates the c estimate matrix for the ad hoc solution
C_ad_hoc = np.zeros((M,M))

#sets the number of Monte Carlo simulations
num_monte_carlo_runs = 100

print("Theta: ", theta)

#iterates through and gets the R_hat for the N random vector samples
for i in range(num_monte_carlo_runs):

    #gets the X matrix, which is N random vector samples of the distribution
    X = np.transpose(np.random.multivariate_normal(mu, R, size=N))
    #prints the size of X
    #print("X: \n", X)

    #gets the R_hat, and theta_hat via maximum likelihood estimation
    #from the X samples of vectors
    R_hat_i, theta_hat_i = ml_estimator(X)

    #gets the current theta_error, which is theta hat minus theta
    theta_error_i = theta_hat_i - theta


    #print("Theta_hat: ", theta_hat_i)
    #print("Theta error hat: ", theta_error_i)

    #adds another error covariance to C
    C = C + np.outer(theta_error_i, theta_error_i)

    #gets the theta hat from the ad hoc solution
    theta_hat_ad_hoc = ad_hoc_estimator(X)


    #gets the theta error_ad_hoc
    theta_error_ad_hoc = theta_hat_ad_hoc - theta

    #print("Teta hat ad hoc: ", theta_hat_ad_hoc)
    #print("Theta error ad hoc: ", theta_error_ad_hoc)

    #adds another error covariance to C_ad_hoc
    C_ad_hoc = C_ad_hoc + np.outer(theta_error_ad_hoc, theta_error_ad_hoc)


print("Theta actual: ", theta)
print()


#then normalizes C by the number of monte carlo runs
C = (1/num_monte_carlo_runs)*C

#then normalizes the C_ad_hoc by the number of monte carlo runs
C_ad_hoc = (1/num_monte_carlo_runs)*C_ad_hoc
print("C ad hoc: \n", C_ad_hoc)


C_ad_hoc_diag = np.diag(C_ad_hoc)
print("C ad hoc diag: \n", C_ad_hoc_diag)

CRB_diag = np.diag(CRB)
print("CRB lower bound: ", CRB_diag)





#end section 3
########################################################################################

########################################################################################
#section 4
#Section 4 includes the plots to compare the actual sample estimation error variance




########################################################################################

# %%
