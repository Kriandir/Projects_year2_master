import numpy as np
from scipy import linalg as sl
import matplotlib.pyplot as plt


x = np.array([1.02,0.95,0.87,0.77,0.67,0.56,0.44,0.30,0.16,0.01])
y = np.array([0.39,0.32,0.27,0.22,0.18,0.15,0.13,0.12,0.13,0.15])
x0 = np.ones(np.shape(x))
b = x**2
print b

def Cholesko(M,b):
    A = np.dot(M.T,M)
    L = sl.cholesky(A, lower=True)
    Ltrans = sl.cholesky(A)
    y = sl.solve_triangular(L,np.dot(M.T,b),lower=True)
    x = sl.solve_triangular(Ltrans,y,lower=False)

    return x

def SVDizzle(M,b):


def Orbitals(x0,x,y,b):
    ysq = y**2
    xy = x*y
    M = np.column_stack((x0,x,xy,y,ysq))
    xy = Cholesko(M,b)
    return xy

xy = Orbitals(x0,x,y,b)
print xy
plt.scatter(x,y, color='r')
plt.plot(x,xy[0] + xy[1]*x+xy[2]*x*y+xy[3]*y+xy[4]*y**2, label = "Linear")
plt.show()

for i in range(len(x)):
    x[i] = x[i] + np.random.uniform(-0.005,0.005)
    y[i] = y[i] + np.random.uniform(-0.005,0.005)

xy = Orbitals(x0,x,y,b)
print xy
plt.scatter(x,y, color='r')
plt.plot(x,xy[0] + xy[1]*x+xy[2]*x*y+xy[3]*y+xy[4]*y**2, label = "Linear")
plt.show()
