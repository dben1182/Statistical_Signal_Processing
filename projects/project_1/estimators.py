#This file defines some estimators

#imports numpy
import numpy as np

A = np.array([[4, 5, 6],
              [2, 2, 2]])

print(np.shape(A)[1])
#implements the maximum likelihood estimator for theta
#takes in as an argument a list of sample vectors from the multivariate
#normal distribution 

#returns R_hat, and the extracted theta_hat
def ml_estimator(X):
    #iterates through and gets each respective individual sample vector

    #gets the shape of X. We will be extracting column vectors
    X_shape = np.shape(X)

    #sets the number of sample vectors
    numSamples = X_shape[1]

    #gets the outer product of X with itself.
    X_transpose = np.transpose(X)

    #gets R_hat
    R_hat = (1/numSamples) * X @ X_transpose

    #gets theta_hat, as the first column of R
    theta_hat = R_hat[:,0] 

    return R_hat, theta_hat


#defines the ad hoc estimator of theta
#estimates theta by summing and averaging over each diagonal of R_hat
def ad_hoc_estimator(X):

    #gets the shape of X
    X_shape = np.shape(X)

    #gets the length of theta, which is the length of a column of X
    theta_length = X_shape[0]
    
    #gets the number of X samples we have
    numSamples = X_shape[1]

    #gets the transposed X matrix
    X_transposed = np.transpose(X)

    #gets R_hat
    R_hat = (1/numSamples) * X @ X_transposed

    #creates theta vector
    theta_hat = np.zeros(theta_length)

    #creates summing variable to sum along diagonals
    sum = 0.0


    #iterates through to estimate each sample of theta
    for i in range(theta_length):
        #resets the sum to zero
        sum = 0.0
        #gets the sum of the diagonal
        for j in range(theta_length - i):
            #accesses the values placed on the diagonals
            sum = sum + R_hat[i + j][j]

        #sets the value into theta
        theta_hat[i] = (1/(theta_length - i))*sum

    #returns theta_hat
    return theta_hat
