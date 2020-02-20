# Import necessary modules
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from scipy.stats import gaussian_kde as kde

from random import seed

# Seed all values to zero
seed(0)
np.random.seed(0)

# Number of dimensions
D = 2
L = 1
K = 1000

# Average and standard deviation
mu_0 = 0
sigma_0 = 1

A_DL = sigma_0 * np.random.randn(D, L) + mu_0;
sigma_vector = sigma_0 * np.random.randn(D, 1) + mu_0

Y = sigma_0 * np.random.randn(L, 1) + mu_0
Z = np.zeros(shape=(D, 1))
Z[0, 0] = sigma_vector[0, 0] * np.random.randn() + mu_0

X = np.dot(A_DL, Y) + Z
X_list = X
X_cov = np.cov(X.T)

print("A vector is")
print(A_DL)
print("Sigma vector is")
print(sigma_vector)

##print("X vector is")
##print(X)
##print("\nMean value of X is")
##print(np.mean(X))
##print("\nCovariance of X is")
##print(X_cov)

for k in range(1, K):
    Z = np.zeros(shape=(D, 1))
    for index in range(1, D):
        Z[index, 0] = sigma_vector[index, 0] * np.random.randn() + mu_0

    Y = sigma_0 * np.random.randn(L, 1) + mu_0
    X = np.dot(A_DL, Y) + Z
##    X = np.dot(A_DL, Y) + Z
##    print("X vector is")
##    print(X)
##    print(np.shape(X))
    X_list = np.append(X_list, X, 1)

    if(k % 1000 == 0):
        print("K = ", k)
##    print(np.shape(X_list))
##    X_list.append(X)
##    X_cov = np.cov(X.T)
##    print("X vector is")
##    print(X)
##    print("\nMean value of X is")
##    print(np.mean(X))
##    print("\nCovariance of X is")
##    print(X_cov)
        
if("D == 2"):
    print("Attempting to Plot...")
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


co_Z = np.zeros(shape=(D,D))
for i in range(0,D):
    co_Z[i, i] = sigma_vector[i]

print("Sigma Z is ")
print(co_Z)
out = np.dot(A_DL, np.transpose(A_DL)) + co_Z
print("Resulting A * A.T + S_Z")
print(out)

print("X combined is")
print(X_list)
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
