#Project 2. 
#This is the project Implementation for project 2 of the class.
#Here, we will be implementing the kalman filter.

#imports the needed libraries
import numpy as np
import csv


#writes down the constants from the mat file
#sets N as 250
N = 250
#sets sigma_n as 10
sigma_n = 10
#sets sigma_u
sigma_u = 0.01
#sets T to 1
T = 1

#reads in s
with open('s.csv', 'r') as file:
    reader = csv.reader(file)
    s = np.array(list(reader)).astype(np.float_)

print(s)


#reads in y_pq
with open 
