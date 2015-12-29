import numpy as np


def compute(a, r):
    U, s, V = np.linalg.svd(a, full_matrices=False)
    S = np.diag(s)
    S = S[:r, :r]
    U = U[:, [i for i in range(r)]]
    V = V[[i for i in range(r)], :]
    return (np.dot(U, S)), (np.dot(S, V))

#Usage:

# a = np.genfromtxt('mat.txt', delimiter=' ')
# U_, V_ = compute(a, 2)
# print(U_)
# print(V_)
