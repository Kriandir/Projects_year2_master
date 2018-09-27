import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as sl

t = np.array([3,11,29,32,47,63,73,99])
w = np.array([74,72,53,35,37,20,18,19])
t0 = np.ones(np.shape(t))

# Perform Cholesko factorization and calculate the solutions following example 3.3 in the book
def Cholesko(M,w):
    A = np.dot(M.T,M)
    L = sl.cholesky(A, lower=True)
    Ltrans = sl.cholesky(A)
    cond = np.linalg.cond(A)
    y = sl.solve_triangular(L,np.dot(M.T,w),lower=True)
    x = sl.solve_triangular(Ltrans,y,lower=False)
    return x,cond

# define linear inputs
def Linear(t0,t,w):
    M = np.column_stack((t0,t))
    x,cond = Cholesko(M,w)
    cond2 = np.linalg.cond(M)
    return x[0],x[1],cond,cond2

# define quadratic inputs
def Quadratic(t0,t,w):
    t1 = t**2
    M = np.column_stack((t0,t,t1))
    x,cond = Cholesko(M,w)
    return x[0],x[1],x[2],cond

# define cubic inputs
def Cubic(t0,t,w):
    t1 = t**2
    t2 = t**3
    M = np.column_stack((t0,t,t1,t2))
    x,cond = Cholesko(M,w)
    return x[0],x[1],x[2],x[3],cond


b,x,cond1,condo = Linear(t0,t,w)
plt.scatter(t,w, color='r')
plt.plot(t,x*t+b, label = "Linear")

b,x,x1,cond2 = Quadratic(t0,t,w)
plt.plot(t,x*t + b + x1*t**2, label = "Quadratic")

b,x,x1,x2,cond3 = Cubic(t0,t,w)
plt.plot(t,b+x*t+x1*t**2+x2*t**3,label = "Cubic")

print '%0.4f & %0.4f & %0.4f ' %(cond1,cond2,cond3)
print condo**2
plt.xlabel("T",fontsize=20)
plt.ylabel("W",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Excercise 1",fontsize=30)
plt.tight_layout()


plt.show()
