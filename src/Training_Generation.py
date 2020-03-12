#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates and Analyzes Training Data


    PART 1: GENERATION
    
    We generate N Gaussian vectors of dimension D whose mean vector is zero,
    and covariance matrix is A * A.T + Σz, where Σz is a diagonal matrix
    whose diagonal entries are σ(z,i)².

    The Gaussian vectors are created from the formula X = A*Y+Z.
        - X is the Gaussian Vector (size D,1) to be generated
        
        - A is the matrix (size D,L) generated with each value independently
            following N(0,1). This matrix will remain constant throughout the
            entire generation of the X vectors.
            
        - Y is the column vector (Size L,1) generated with each value
            independently following N(0,1). The Y vector will be regenerated
            each time we want to generate a new X vector.
            
        - Z is the column vector (Size D,1) generated with each value
            independently following N(0,σ(z,i)²). Each value of σ(z,i)² itself
            independently folows N(0,1). The values of σ(z,i)² will remain
            constant throughout the entire generation of the X vectors. The
            Z vector will be regenerated each time we want to generate a new
            X vector.
    Notes:
        - L << D: The latent vector size should be much smaller than the size
            of the explicit vector size.
"""


# Import necessary modules
from random import seed

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde as kde

def generate_Z(sigmaVector):
    Z = np.zeros(shape=(sigmaVector.shape[0], 1))
    for index in range(1, D):
        Z[index, 0] = sigmaVector[index, 0] * np.random.randn() + 0
    return Z

def generate_A(rows, columns):
    return generate_matrix(0, 1, rows, columns)

def generate_Y(size):
    return generate_matrix(0, 1, size, 1)

def generate_Sigma(size):
    return generate_matrix(0, 1, size, 1)

def generate_matrix(mean, variance, rows, columns):
    return variance * np.random.randn(rows, columns) + mean

# Seed all values to zero
seed(0)
np.random.seed(0)

# Number of dimensions
D, L, N = 2, 1, 1000
draw = True

A = generate_A(D, L)
sigma = generate_Sigma(D)
Y = generate_Sigma(L)
X_List = []

for n in range(N):
    Z = generate_Z(sigma)
    X = np.dot(A, Y) + Z
    X_List.append(X)

X_list = np.array(X_List)
#
print(X_list)
print(X_list.shape)
X_list = np.reshape(X_list, (N, D))
print(X_list)
print(X_list.shape)



# Average and standard deviation
##mu_0 = 0
##sigma_0 = 1
##
##A_DL = sigma_0 * np.random.randn(D, L) + mu_0;
##sigma_vector = sigma_0 * np.random.randn(D, 1) + mu_0
##
##Y = sigma_0 * np.random.randn(L, 1) + mu_0
##Z = np.zeros(shape=(D, 1))
##Z[0, 0] = sigma_vector[0, 0] * np.random.randn() + mu_0
##
##X = np.dot(A_DL, Y) + Z
##X_list = X
##X_cov = np.cov(X.T)

##print("A vector is")
##print(A_DL)
##print("Sigma vector is")
##print(sigma_vector)

##print("X vector is")
##print(X)
##print("\nMean value of X is")
##print(np.mean(X))
##print("\nCovariance of X is")
##print(X_cov)

##for k in range(1, N):
##    Z = np.zeros(shape=(D, 1))
##    for index in range(1, D):
##        Z[index, 0] = sigma_vector[index, 0] * np.random.randn() + mu_0
##
##    Y = sigma_0 * np.random.randn(L, 1) + mu_0
##    X = np.dot(A_DL, Y) + Z
##    X = np.dot(A_DL, Y) + Z
##    print("X vector is")
##    print(X)
##    print(np.shape(X))
##    X_list = np.append(X_list, X, 1)
##
##    if(k % 1000 == 0):
##        print("K = ", k)
##    print(np.shape(X_list))
##    np.append(X_list, X)
##    X_cov = np.cov(X.T)
##    print("X vector is")
##    print(X)
##    print("\nMean value of X is")
##    print(np.mean(X))
##    print("\nCovariance of X is")
##    print(X_cov)

##print(X_list.shape)
if draw:
    if D == 2:
        print("Attempting to Plot 2D")
        plt.xlabel("X1")
        plt.ylabel("X2")
        densObj = kde( X_list )

        def makeColours( vals ):
            colours = np.zeros( (len(vals),3) )
            norm = Normalize( vmin=vals.min(), vmax=vals.max() )

            #Can put any colormap you like here.
            colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

            return colours
        colours = makeColours( densObj.evaluate( X_list ) )
        title = "L = " + str(L) + " K = " + str(K)
        plt.scatter(X_list[0], X_list[1], color=colours)
        plt.title(title)
        plt.show(block=False)
        
        print("Average X1-value: ", np.mean(X_list[0]))
        print("Average X2-value: ", np.mean(X_list[1]))

    if D == 3:
        print("Attempting to Plot 3D")
        def scatter3d(x,y,z, cs, colorsMap='jet'):
            cmx = plt.get_cmap(colorsMap)
            cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmx)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
            scalarMap.set_array(cs)
            fig.colorbar(scalarMap)
            plt.show(block=False)
        scatter3d(X_list[0], X_list[1], X_list[2])

##co_Z = np.zeros(shape=(D,D))
##for i in range(0,D):
##    co_Z[i, i] = sigma_vector[i] * sigma_vector[i]
##
##print("Sigma Z is ")
##print(co_Z)
##out = np.dot(A_DL, np.transpose(A_DL)) + co_Z
##
##print("X combined is")
##print(X_list)

def covert(X_list):
    total = np.zeros(shape=(D,D))
    for i in range(0, k):
        total += (X_list[:,i] - np.mean(X_list[:,i]) * np.transpose(X_list[:,i] - np.mean(X_list[:,i])))
        ##print(total)
    total /= (float(k) - 1)
    return total;


print("\nResulting A * A.T + S_Z")
print(out)
print("\nMystefied cov")
print(covert(X_list))
print("\nCalculated cov")
print(np.cov(X_list, rowvar=False))



##    saveBool = input("Save image (Y/n)? ")
##    if(saveBool):
##        name = input("Name: ")
##        plt.savefig(name)

##print("X_list is")
##print(X_list)
##print("X1 values in X_List")
##print(X_list[0])
##print("X2 values in X_List")
##print(X_list[1])
##
##
##X_cov = np.cov(X_list)
##print("\nCovariance of X is")
##print(X_cov)
##print(np.shape(X_cov))
##print(np.mean(X_cov, axis=1))
##
##
##Z_Diag = np.zeros(shape=(D,D))
##for index in range(0, D):
##    Z_Diag[index, index] = Z[index, 0]
##
##Alt_cov = np.dot(A_DL, A_DL.T) + Z_Diag
##print("\nAlt Cov is")
##print(Alt_cov)
##print(np.shape(Alt_cov))
##print("\nAlt mean vector is")
##print(np.mean(Alt_cov, axis=1))
