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
import math

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde as kde


def generateIndependentFollowing(mean, variance, rows, columns):
    """
    Returns a numpy array following the Gaussian N(mean, variance)
    :param mean: mean of the Guassian to be sampled from
    :param variance: variance of the Gaussian to be sampled from
    :param rows: number of rows
    :param columns: number of columns
    :return numpy array
    """
    return math.sqrt(abs(variance)) * np.random.randn(rows, columns) + mean


def generateZ(sigmaVec):
    """
    Returns D entries of D independent Gaussian following a normal distribution
    mean = 0 and whose standard deviation is a parameter.
    :param sigmaVec: Standard deviations following Gaussian N(0,1)
    :return: numpy array
    """
    Z = abs(sigmaVec) * np.random.randn(sigmaVec.shape[0], 1)
    return Z


def generateA(row, column):
    """
    Returns a vector of size D x L following the Gaussian N(0,1)
    :param row: Number of rows (D)
    :param column: Number of columns (L)
    :return: numpy array
    """
    return generateIndependentFollowing(mean=0, variance=1, rows=row, columns=column)


def generateY(size):
    """
    Returns a vector of size L following Gaussian N(0,1)
    :param size: Number of rows (L)
    :return: numpy array
    """
    return generateIndependentFollowing(mean=0, variance=1, rows=size, columns=1)


def generateSigma(size):
    """
    Returns a vector of standard deviations following Gaussian N(0,1)
    :param size: Size of the sigma vector to be generated
    :return: numpy array
    """
    return generateIndependentFollowing(mean=0, variance=1, rows=size, columns=1)


def plotDensity2D(coordinates, L):
    """
    Plots a 2D density plot
    :param coordinates: 2D numpy array
    :param L: Size of latent vector
    :return: None
    """
    plt.figure(-1)
    plt.xlabel("X1")
    plt.ylabel("X2")
    densobj = kde(coordinates)

    def makeColors(vals):
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )
        return [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba(val) for val in vals]

    colors = makeColors(densobj.evaluate(coordinates))
    title = "L = " + str(L) + " D = " + str(coordinates.shape[0]) + " N = " + str(coordinates.shape[1])
    
    plt.scatter(coordinates[0], coordinates[1], color=colors)
    plt.title(title)
    plt.show(block=False)
    
    print("Average X1-value: ", np.mean(coordinates[0]))
    print("Average X2-value: ", np.mean(coordinates[1]))


def generateCovarianceMatrix(A, Sigma):
    """
    Generates a covariance matrix using formula A * A.T + Sigma_Z, with Sigma_Z
    being a diagonal matrix with the genrated Sigma values squared.
    :param A: Matrix to generate covariance from
    :param Sigma: Vector of standard deviations
    :return: Numpy array
    """
    co_Z = np.diag(np.ndarray.flatten(np.square(Sigma)))
    return np.dot(A, np.transpose(A)) + co_Z


def plotHistogram2D(matrix):
    counter = 1
    for row in matrix:
        print(row)
        plt.figure(counter)
        _ = plt.hist(row, bins='auto')
        title = "Histogram for X" + str(counter)
        plt.title(title)
        plt.show(block=False)
        counter += 1


def main():
    # Seed all values to zero
    seed(100)
    np.random.seed(100)

    # Number of dimensions
    D = 2
    L = 1
    N = 100

    Sigma = generateSigma(D)
    print(Sigma)
    # print(np.mean(Sigma))

    A = generateA(D, L)
    print(np.mean(A))

    xList = []

    covMatrix = generateCovarianceMatrix(A, Sigma)
    print(covMatrix)
    # Come up with estimate of covariance matrix and mean

    xListp = []

    meanVec = np.zeros(shape=(D,))
        
    # Generate for each n value
    for n in range(N):
        Z = generateZ(Sigma)
        Y = generateY(L)
        X = np.dot(A, Y) + Z
        xList.append(X)
        arrGen = np.random.multivariate_normal(mean=meanVec, cov=covMatrix)
        xListp.append(arrGen)

    if D != 2:
        xListpM = np.squeeze(np.array(xListp)).T
        print("Shape:\n", xListpM.shape)

        plotDensity2D(xListpM, L)
        plotHistogram2D(xListpM)

#    eigVals, eigVecs = np.linalg.eig(out)
#    print("Eigenvalues \n", eigVals)
#    print("Eigenvectors \n", eigVecs)
#    cos , -sin
#    sin, cos

#    vector = eigVecs[1]
#    theta = math.atan(vector[0]/vector[1])
#    print(theta)

    if D == 2:
        # Reshape xList to form a 2D matrix
        xMatrix = np.squeeze(np.array(xList)).T

        """Alternatives:    
        np.squeeze(np.stack(xList, axis=1))
        np.squeeze(np.stack(xList)).T
        """

        # Plot density
        plotDensity2D(xMatrix, L)

        # Plot histogram
        plotHistogram2D(xMatrix)


if __name__ == '__main__':
    main()
