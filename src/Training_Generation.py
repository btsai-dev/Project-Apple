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

# Seed all values to zero
seed(0)
np.random.seed(0)

# Number of dimensions
D = 2
L = 1
N = 10000

def generateIndependentFollowing(mean, variance, rows, columns):
    return variance * np.random.randn(rows, columns) + mean

def generateZ(sigmaVector):
    rows = sigmaVector.shape[0]
    Z = np.zeros( shape=( rows, 1 ) )
    
    for index in range( 0, rows ):
        Z[index, 0] = sigmaVector[index, 0] * np.random.randn() + 0
    return Z

def generateA(row, column):
    return generateIndependentFollowing( mean=0, variance=1, rows=row, columns=column)

def generateY(size):
    return generateIndependentFollowing( mean=0, variance=1, rows=size, columns=1 )

def generateSigma(size):
    return generateIndependentFollowing( mean=0, variance=1, rows=size, columns=1 )

def plot2D(coordinates, L):
    ##print("Attempting to Plot...")
    
    plt.figure(-1)
    plt.xlabel("X1")
    plt.ylabel("X2")
    densObj = kde( coordinates )

    def makeColours( vals ):
        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )

        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

        return colours
    colours = makeColours( densObj.evaluate( coordinates ) )
    title = "L = " + str(L) + " N = " + str( coordinates.shape[0])
    
    plt.scatter(coordinates[0], coordinates[1], color=colours)
    plt.title(title)
    plt.show(block=False)
    
    print("Average X1-value: ", np.mean(xList[0]))
    print("Average X2-value: ", np.mean(xList[1]))
    
Sigma = generateSigma( D )
print(np.mean(Sigma))

A = generateA( D, L )
print(np.mean(A))


xArrayList = np.zeros( shape=( 2, 1 ) )
xList = []

for n in range( N ):
    Z = generateZ( Sigma )
    Y = generateY( L )
    X = np.dot( A, Y ) + Z
    xList.append(X)
    
xMatrix = None

if (D == 2):
    # Reshape xList to form a 2D matrix
    xMatrix = np.squeeze(np.array(xList)).T

    # Plot density
    plot2D(xMatrix, L)
    
    """Alternatives:

    np.squeeze(np.stack(xList, axis=1))
    np.squeeze(np.stack(xList)).T
    """

    # Plot histogram
    counter = 0
    for row in xMatrix:
        print(row)
        plt.figure(counter)
        _ = plt.hist(row, bins='auto')
        title = "Histogram for X" + str(counter)
        plt.title(title)
        plt.show(block=False)
        counter += 1
        

    
    
