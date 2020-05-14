#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates and Analyzes Training Data
    PART 1: GENERATION
    
    We generate N Gaussian vectors of dimension D whose mean vector is zero,
    and covariance matrix is A * A.T + ΣZ, where ΣZ is a diagonal matrix
    whose diagonal entries are σ(z,i)².
    

    GENERATION TYPE ALPHA
    
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
            

    GENERATION TYPE BRAVO

    The covariance matrix and vector mean of the data generated from GENERATION
    TYPE ALPHA are computed. A new set of data is generated from the covariance
    matrix and vector mean.


    GENERATION TYPE CHARLIE

    The covariance matrix is computed from A * A.T + Σz as stated above. A new
    set of data is generated from this analytically-derived covariance matrix
    and a zero vector mean.


    PART 2: COMPARISONS

    The Frobenius Norm can be used to compare covariance matrices, and R
    integration was implemented to utilize a multivariate Kolmogorov-Smirnov
    Test of Means. Online documentation is outdated, so details on the function
    implementation should be found by running help(KStest) or ?KStest in the
    R interactive environment.

    Only the multivariate KS Test of Means requires the downloading and the
    installation of the R language on your system. The Frobenius Norm does not
    require the R language. 
"""

# Import necessary modules
import random
import traceback
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde as kde


# Global constants
PLOTTING = True
R_INTEGRATION = False


def getRunningR():
    if R_INTEGRATION:
        r = None
        try:
            import pyper as pr
            r = pr.R()
            r.has_numpy = True
            r.has_pandas = False
            r.run('if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
            r('BiocManager::install("GSAR")')
            r('library(GSAR)')
            r('library(MASS)')
            print("Loaded libraries")
            return r
        except ImportError:
            print("Import error, likely pyper not installed.")
            traceback.print_exc()
        except:
            print("Some other error.")
            traceback.print_exc()
        finally:
            return r
    else:
        print("R_INTEGRATION is set to false.")
        return None
    

def calculateKSTest(r, dataset):
    if R_INTEGRATION:
        # print(dataset)
        # print(dataset.shape[0])
        # print(dataset.shape[1])
        r.assign('dataset', dataset)
        r.assign('columns', dataset.shape[1])
        print("Calculating, please wait.")
        print(r('result <- KStest(object=dataset, group=c( rep(1,columns/2),rep(2,columns/2) ), pvalue.only=FALSE)'))
        print(r('result$p.value'))
        print(r('result$statistic'))
        return r.get('result$p.value'), r.get('result$statistic')
    else:
        print("R_INTEGRATION is set to false.")
        return None


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


def retrieveScale(coordinates):
    return math.ceil(max(np.amax(coordinates), abs(np.amin(coordinates))))


def plotDensity2D(coordinates, title="2D Graph"):
    """
    Plots a 2D density plot
    :param coordinates: 2D numpy array
    :param title: Title of graph
    :return: None
    """
    plt.figure()
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    densobj = kde(coordinates)

    def makeColors(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        return [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]

    colors = makeColors(densobj.evaluate(coordinates))

    plt.scatter(coordinates[0], coordinates[1], color=colors)

    scale = retrieveScale(coordinates)
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(title)
    plt.show(block=False)


def plotDensity3D(coordinates, title="3D Graph"):
    """
    Plots a 3D density plot
    :param coordinates: 3D numpy array
    :param title: Title of graph
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter(coordinates[0], coordinates[1], coordinates[2])
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('$X_3$')
    plt.title = title
    plt.show(block=False)


def plotHistograms(realizations):
    """
    Plots histograms from realizations. D histograms will be plotted.
    :param realizations: D x N numpy array
    :return: None
    """
    counter = 1
    for variable in realizations:
        plt.figure()
        _ = plt.hist(variable, bins='auto')
        title = "$X_{var:d}$ histogram"
        plt.title(title.format(var=counter))
        scale = retrieveScale(realizations)

        plt.xlim(-scale, scale)

        plt.show(block=False)
        counter += 1


def plotHistogram2D(realizations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    hist, xedges, yedges = np.histogram2d(realizations[0], realizations[1])
    plt.title("3D histogram of 2D normally distributed data points")
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    # Construct arrays for the anchor positions of the bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    scale = retrieveScale(realizations)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=np.random.rand(3,),
             zsort='average',
             edgecolor='k',
             linewidth=0.5)

    plt.show(block=False)


def getRotationFromCov(cov, radian=True):
    """
    Calculates the rotation angle
    :param cov: Covariance matrix to be used
    :param radian: Determines if return value should be in radians or degrees
    :return: Angle in radians or degrees
    """
    eigVals, eigVecs = np.linalg.eig(cov)
    rotationMatrix = np.dot(np.diag(np.ndarray.flatten(eigVals)), np.linalg.inv(cov))
    vector = rotationMatrix[1]

    if radian:
        return math.atan(vector[0]/vector[1])
    else:
        return math.atan(vector[0]/vector[1]) * 180 / math.pi


def theoreticalCovMatrix(A, Sigma):
    """
    Generates a covariance matrix using formula A * A.T + Sigma_Z, with Sigma_Z
    being a diagonal matrix with the genrated Sigma values squared.
    :param A: Matrix to generate covariance from
    :param Sigma: Vector of standard deviations
    :return: Numpy array
    """
    co_Z = np.diag(np.ndarray.flatten(np.square(Sigma)))
    return np.dot(A, np.transpose(A)) + co_Z


def empiricalCovMatrix(data):
    dim, size = data.shape
    meanVec = np.mean(data, axis=1, dtype=np.float64).reshape(-1, 1)

    cov = np.zeros(shape=(dim, dim), dtype=np.float64)
    for k in range(size):
        cov += np.dot(data[:, k].reshape(-1, 1) - meanVec, (data[:, k].reshape(-1, 1) - meanVec).T)
    return cov * (1 / (size - 1))


def covEigenValues(cov):
    return np.linalg.eig(cov)[0]


def calcFrobeniusNorm(theoretical, empirical):
    return np.linalg.norm(empirical - theoretical, ord='fro')


def getMeanVector(matrix):
    return np.mean(matrix, axis=1)


def resultsAlpha(Sigma, A, D, L, N):
    """
    Generates reuslts according to the definition described in GENERATION TYPE ALPHA.
    :param Sigma:
    :param A:
    :param D: Input and output dimensions
    :param L: Latent dimensions
    :param N: Number of realizations
    :return: Numpy array
    """
    

    listA = []

    # Generate for each n value
    for n in range(N):
        Z = generateZ(Sigma)
        Y = generateY(L)
        X = np.dot(A, Y) + Z
        listA.append(X)
        
    listA = np.squeeze(np.array(listA)).T
    return listA


def resultsBravo(listA, D, N):
    """
    Generates reuslts according to the definition described in GENERATION TYPE BRAVO.
    :param listA: Data generated according to GENERATION TYPE ALPHA
    :param D: Input and output dimensions
    :param N: Number of realizations
    :return: Numpy array
    """
    # Derive covariance matrix and vector mean from GENERATION ALPHA
    covEmpirical = empiricalCovMatrix(listA)
    covMean = np.reshape(getMeanVector(listA), newshape=(D,))
    listB = []
    for n in range(N):
        arrGen = np.random.multivariate_normal(mean=covMean, cov=covEmpirical)  
        listB.append(arrGen)
    listB = np.squeeze(np.array(listB)).T
    return listB


def resultsCharlie(Sigma, A, D, N):
    """
    Generates reuslts according to the definition described in GENERATION TYPE CHARLIE.
    :param Sigma:
    :param A:
    :param D: Input and output dimensions
    :param N: Number of realizations
    :return: Numpy array
    """
    listC = []
    covTheoretical = theoreticalCovMatrix(A, Sigma)
    for n in range(N):
        arrGen = np.random.multivariate_normal(mean=np.zeros(shape=(D,)), cov=covTheoretical)
        listC.append(arrGen)
    listC = np.squeeze(np.array(listC)).T
    return listC

def main():
    # Seed all values to zero
    random.seed(0)
    np.random.seed(0)


    # Debugging Parameters
    
    # Enables or Disables R integration
    
    

    # Number of dimensions
    D = 2
    L = 1
    N = 500

    Sigma = generateSigma(D)
    A = generateA(D, L)

    listA = resultsAlpha(Sigma, A, D, L, N)
    listB = resultsBravo(listA, D, N)
    listC = resultsCharlie(Sigma, A, D, N)
    
    
    # Generate theoretical data from zero mean and cov (A * A.T + sigma_Z)

    if PLOTTING:
        if D == 2:
            plotDensity2D(listA, title="Alpha Plotting")
            plotDensity2D(listB, title="Bravo Plotting")
            plotDensity2D(listC, title="Charlie Plotting")
            # plotHistogram2D(xList)
            # plotHistograms(xList)

        if D == 3:
            plotDensity3D(listA, title="Alpha Plotting")
            plotDensity3D(listB, title="Bravo Plotting")
            plotDensity3D(listC, title="Charlie Plotting")
            # plotHistograms(xList)
            # plotDensity3D(xListp, title="Generated from Covariance Matrix")
            # plotHistograms(xListp)

#    print("Theoretical Cov matrix:\n", covTheoretical)
#    print("Empirical Cov matrix:\n", covEmpirical)
#    print("Theoretical Angle (Deg):\n", getRotationFromCov(covTheoretical, radian=False))
#    print("Empirical Angle (Deg):\n", getRotationFromCov(covEmpirical, radian=False))
#    print("Theoretical Eigenvalues:\n", covEigenValues(covTheoretical))
#    print("Theoretical Eigenvalues:\n", covEigenValues(covEmpirical))
#    print("Frobenius Norm:\n", calcFrobeniusNorm(covTheoretical, covEmpirical))

    if R_INTEGRATION:
        xList_Em_Th = np.concatenate( (xListn, xListp), axis=1 )
        xList_Em_Ay = np.concatenate( (xListn, xList), axis=1 )
        xList_Th_Ay = np.concatenate( (xListp, xList), axis=1 )
        print("Getting a working R instance")
        r = getRunningR()
        print("Calculating KS Test")
        KS_pvalueET, KS_statisticET = calculateKSTest(r, xList_Em_Th)
        KS_pvalueEA, KS_statisticEA = calculateKSTest(r, xList_Em_Ay)
        KS_pvalueTA, KS_statisticTA = calculateKSTest(r, xList_Th_Ay)
        print("KS p-value for Empirical and Theoretical:", KS_pvalueET)
        print("Test statistic for Empirical and Theoretical:", KS_statisticET)
        print("KS p-value for Empirical and AY+Z:", KS_pvalueEA)
        print("Test statistic for Empirical and AY+Z:", KS_statisticEA)
        print("KS p-value for AY+Z and Theoretical:", KS_pvalueTA)  
        print("Test statistic for AY+Z and Theoretical:", KS_statisticTA)
    
    



if __name__ == '__main__':
    main()
