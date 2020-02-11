import numpy as np

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
Z = np.asmatrix( np.zeros(shape=(D, 1)) ,dtype=np.longdouble )
for index in range(0, D):
    Z[index, 0] = sigma_vector[index, 0] * np.random.randn() + mu_0

print()
print("Z Vector")
print(Z)

X_list = []
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
