# EXCERCISE 4

import numpy as np
import matplotlib.pyplot as plt

# Create Hilbert matrix
def CreateMatrix(n):
    Matrix = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            Matrix[i][j] = 1./((i+1)+(j+1)-1)
    return Matrix

# Generate vector of ones
def GenX(n):
    x = np.ones((n))
    return x

# get solution and xhat for the matrix and x
def SolveMatrix(n):
    x = GenX(n)
    M = CreateMatrix(n)
    b = np.dot(M,x)
    xhat = np.linalg.solve(M,b)
    return b,xhat,M,x

# calculate the residual and the relative error and take the inf norms of them.
# Also calculate the norm of the condition number
def NormResidual(n):
    b,xhat,M,x = SolveMatrix(n)
    residual = b-np.dot(M,xhat)
    r = np.linalg.norm(residual,ord=np.inf)
    relerror = (((xhat-x)*100)/x)
    errorx = np.linalg.norm(relerror,ord =np.inf)
    cond = np.linalg.cond(M)
    return r,errorx,cond

# define lists and amount of arrays of size nxn
nsize = 26
xlist = []
Reslist = []
Errlist = []
Condloglist = []
Condlist = []
for i in range(2,nsize):
    r,errorx,cond = NormResidual(i)
    xlist.append(i)
    Reslist.append(r)
    Errlist.append(errorx)
    Condlist.append(cond)
    # take the log10 of the condition number in order to find the amount of digits that we lose
    Condloglist.append(np.log10(cond))

# plot everything
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True)
plt.suptitle("Excercise 2.6",fontsize = 30)
ax1.plot(xlist,Reslist)
ax1.tick_params(axis='both', labelsize=17)
ax1.set_ylabel('Residuals', fontsize = 15)
ax2.plot(xlist,Errlist)
ax2.set_ylabel("Relative Error [%]", fontsize = 15)
ax2.tick_params(axis='both', labelsize=17)
ax2.axhline(100,color='r')
ax3.plot(xlist,Condlist)
ax3.set_ylabel("Condition Number",fontsize = 15)
ax3.set_xlabel("N",fontsize = 20)
ax3.tick_params(axis='both', labelsize=17)
ax4.set_ylabel("Log10(Condition Number)",fontsize = 15)
ax4.set_xlabel("N",fontsize = 20)
ax4.tick_params(axis='both', labelsize=17)
ax4.plot(xlist,Condloglist)
plt.show()
