from scipy import linalg as sl
import numpy as np

A = np.zeros((4,4))
for i in range(np.shape(A)[0]):
    for j in range(np.shape(A)[0]):
        if i == j:
            A[i][j] = 2
        if i == (j-1):
            A[i][j] = 1
        if j == (i-1):
            A[i][j] = 1

P = sl.lu(A)
U = P[2]
L = P[1]
P = P[0]



def CalcInv(A):
    U = np.copy(A)
    I = np.identity(np.shape(A)[0])
    invlist = []
    invlist2 = []
    h = 0
    for i in np.arange(0,np.shape(U)[0]):

        for k in np.arange(i+1,np.shape(U)[0]):
            d = 0
            if U[k][h] == 0:
                continue
            else:
                for g in np.arange(0,np.shape(U)[1]):
                    if U[k][g] == 0:
                        continue

                    if U[k][g] != 0:
                        if d == 0:
                            d = U[k][i]/U[i][i]
                            invlist.append(d)
                        U[k][g] -= d * U[i][g]
        h+=1

    h = -1

    for i in np.arange(-1,-np.shape(U)[0]-1,-1):
        for k in np.arange((i-1),-np.shape(U)[0]-1,-1):
            d = 0

            if U[k][h] == 0:
                continue
            else:
                for g in np.arange(-1,-np.shape(U)[1]-1,-1):
                    if U[k][g] == 0:
                        continue

                    if U[k][g] != 0:
                        if d == 0:
                            d = U[k][i]/U[i][i]
                            invlist2.append(d)
                        U[k][g] -= d * U[i][g]
        h-=1

    for i in range(len(invlist)):
        for j in np.arange(0,np.shape(U)[1]):
            I[i+1][j] -= I[i][j]*invlist[i]

    for i in range(len(invlist2)):
        for j in np.arange(-1,-np.shape(U)[1]-1,-1):
            I[-i-2][j] -= I[-i-1][j]*invlist2[i]

    for i in np.arange(0,np.shape(U)[0]):
        c = 0
        for j in np.arange(0,np.shape(U)[1]):
            if U[i][j] !=0:

                if c == 0:
                    c = U[i][j]
            if c != 0:
                U[i][j] = U[i][j]/c
                I[i][j] = I[i][j]/c

    return I

E = CalcInv(A)
U = CalcInv(U)
L = CalcInv(L)
Z = np.dot(U,L)
print E
print Z
