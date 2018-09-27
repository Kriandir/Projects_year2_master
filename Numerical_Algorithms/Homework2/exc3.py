# EXCERCISE 3

from scipy import linalg as sl
import numpy as np

# Define the matrix A
A = np.zeros((4,4))
for i in range(np.shape(A)[0]):
    for j in range(np.shape(A)[0]):
        if i == j:
            A[i][j] = 2
        if i == (j-1):
            A[i][j] = 1
        if j == (i-1):
            A[i][j] = 1

# get the L and U
P = sl.lu(A)
U = P[2]
L = P[1]
P = P[0]

# Define identity matrix
b = np.identity(np.shape(L)[0])
# Solve L*d = b
d = sl.solve_triangular(L,b,lower=True)

# Solve U*x = d
Ainv = sl.solve_triangular(U,d)
print Ainv
