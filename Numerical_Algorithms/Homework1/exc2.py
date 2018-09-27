import numpy as np
import math
import matplotlib.pyplot as plt

# Function returning F(x)
def Fx(x):
    return(np.exp(-2*x))

# Function returning exact solution
def exactsolution(x):
    return(4*np.exp(-2*x))

# Function returning Centered solution
def cFD(x,h):
    return (Fx(x+h) - 2 * Fx(x) + Fx(x-h))/(h**2)

# Function returning Forward solution
def fFD(x,h):
    return (Fx(x+(2*h)) - 2 * Fx(x + h) + Fx(x))/(h**2)

# Function calculating error Centered
def errorcFD(x,h):
    return( abs(cFD(x,h) - exact(x)))

# Function calculating error Forward
def errorfFD(x,h):
    return( abs(fFD(x,h) - exact(x)))

# Intializing values
x = 0.5
cf = []
ff = []
xlist=[]
exactz = []

# Looping over specified range (1 till 1*10^-16)
for h in range(0,17):
    h = -1*h
    i = 10**h
    cf.append(errorcFD(x,i))
    ff.append(errorfFD(x,i))
    exactz.append(exactsolution(x))
    xlist.append(i)

# Plot everything
plt.plot(xlist,cf, label = "Centered FD")
plt.plot(xlist,ff, label = "Forward FD")
cf = np.array(cf)
ff = np.array(ff)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("h",fontsize=20)
plt.ylabel("Error",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Comparison between Centered FD and Forward FD",fontsize=30)
plt.tight_layout()
plt.show()
