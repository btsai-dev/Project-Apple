import numpy as np

D = 20
L = 10
K = 20
mu_0 = 0
sigma_0 = 1

A_DL = sigma_0 * np.random.randn(D, L) + mu_0;
sigma_vector = sigma_0 * np.random.randn(D, 1) + mu_0



Y = sigma_0 * np.random.randn(L, 1) + mu_0
X = np.dot(A_DL, Y) + Z
X_list = X
# X_cov = np.cov(X.T)
##print("X vector is")
##print(X)
# print("\nMean value of X is")
# print(np.mean(X))
# print("\nCovariance of X is")
# print(X_cov)



for k in range(1, K):
    Z = np.zeros(shape=(D, 1))
    for index in range(0, D):
        Z[index, 0] = sigma_vector[index, 0] * np.random.randn() + mu_0

    Y = sigma_0 * np.random.randn(L, 1) + mu_0
    X = np.dot(A_DL, Y) + Z
##    X = np.dot(A_DL, Y) + Z
##    print("X vector is")
##    print(X)
##    print(np.shape(X))
    X_list = np.append(X_list, X, 1)
##    print(np.shape(X_list))
    # X_list.append(X)
    # X_cov = np.cov(X.T)
    # print("X vector is")
    # print(X)
    # print("\nMean value of X is")
    # print(np.mean(X))
    # print("\nCovariance of X is")
    # print(X_cov)

print("X_list is")
print(X_list)
X_cov = np.cov(X_list)
print("\nCovariance of X is")
print(X_cov)
print(np.shape(X_cov))
print(np.mean(X_cov, axis=1))


Z_Diag = np.zeros(shape=(D,D))
for index in range(0, D):
    Z_Diag[index, index] = Z[index, 0]

Alt_cov = np.dot(A_DL, A_DL.T) + Z_Diag
print("\nAlt Cov is")
print(Alt_cov)
print(np.shape(Alt_cov))
print("\nAlt mean vector is")
print(np.mean(Alt_cov, axis=1))
'''
D = 10
L = 10
K = 1
mu_0 = 0
sigma_0 = 1


# Generating the A_DL Matrix sampled from N(0,1)
A_DL = np.asmatrix( sigma_0 * np.random.randn(D, L) + mu_0, dtype=np.longdouble )

print("A_DL")
print(A_DL)

# Generating a vector full of sigmas sampled from N(0, 1)
sigma_vector = np.asmatrix( sigma_0 * np.random.randn(D, 1) + mu_0, dtype=np.longdouble )

print()
print("Sigma Vector")
print(sigma_vector)

    # Generating a vector sampled from N(0, sigma vector value)
print()
print("Z Vector")
print(Z)
Z_Diag_Cov = np.cov(Z_Diag)

print()
print("Getting diagonal matrix of the Z vector")
print(Z_Diag)
print("Getting covariance matrix of the Z vector")
print(Z_Diag_Cov)

for k in range(0, K):
    # X = np.asmatrix(np.zeros(shape=(D,1)), dtype=np.longdouble)
    
    Y = np.asmatrix( sigma_0 * np.random.randn(L, 1) + mu_0, dtype=np.longdouble )
    print()
    print("Y Vector")
    print(Y)
    X = A_DL * Y + Z
    print()
    print("X Vector")
    print(X)
    X_list.append(X)

    print()
    print("Covariance matrix of X is")
    print(np.cov(X))

    
    Calc = A_DL * A_DL.T + Z_Diag
    print()
    print("Covariance calculation from A * A_T + Z")
    print(Calc)
'''

