import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def compute(a, r):
    U, s, V = np.linalg.svd(a)
    S = np.diag(s)
    S = S[:r, :r]
    U = U[:, [i for i in range(r)]]
    V = V[[i for i in range(r)], :]
    return (np.dot(U, S)), (np.dot(S, V))

#Usage:

a = np.genfromtxt('mat.txt', delimiter=' ')
transformer = TfidfTransformer()
tfdif = transformer.fit_transform(a)
a = tfdif.toarray()
U_, V_ = compute(a,3)
# print(U_)
# print(V_)

