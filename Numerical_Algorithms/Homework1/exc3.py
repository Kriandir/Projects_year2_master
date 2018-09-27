import numpy as np
import math
import matplotlib.pyplot as plt

# Function used with the for loop method
def Frloop (n,a,r):
    return(a*((1.+(r/n))))

# Function used with the exp method
def Fr(n,a,r):
    return (a*np.exp(n*np.log(1.+(r/n))))

# Function used to get single precision with the exp method
def Frfloat32(n,a,r):
    return (np.float32(a)*np.float32(np.exp(np.float32(n)*np.float32(np.log(np.float32(1.)+np.float32((r/n)))))))

# Initializing values
r = 0.05
nsteps = [1,4,12,365]
sollist = []
astepstotal = []
narray = np.logspace(0,20,21)
narray2 = np.float32(np.logspace(0,20,21, dtype=np.float32))
narrayx = np.logspace(0,20,21)

# Function used to loop over the double precision array
def Loopings(a,narray,r):
    for n in np.nditer([narray],op_flags=["readwrite"]):
        n[...] = Fr(n,a,r)

# Function used to loop over the single precision array
def Loopingsfloat32(a,narray,r):
    for n in np.nditer([narray],op_flags=["readwrite"]):
        n[...] = Frfloat32(n,a,r)

# Function used to properly make up the xarray
def xarray(narray,r):
    for n in np.nditer([narray],op_flags=["readwrite"]):
        n[...] = (r/n)

# Get the values for the loop function for the different n values and print it
for i in nsteps:
    solution = 0
    a= 100
    asteps=[]
    for n in range(1,i+1):
        solution = Frloop(i,a,r)
        a = solution
        asteps.append(a)
    astepstotal.append(asteps)
    sollist.append(solution)

for i in range(len(sollist)):
    print "For an amount of steps of %0.f we get a final compound interest of: %0.5f" %(nsteps[i],sollist[i])

# Get values for the exp function for the different n values and print it
exactsolution = []
for i in nsteps:
    a = 100
    exactsolution.append(Fr(i,a,r))

for i in range(len(exactsolution)):
    print "For an amount of steps of %0.f we get a final compound interest of: %0.5f" %(nsteps[i],exactsolution[i])

# SINGLE VS DOUBLE PRECISION
xarray(narrayx,r)

a = 100
Loopings(a,narray,r)

r = np.float32(r)
a = 100
a = np.float32(a)
Loopingsfloat32(a,narray2,r)

# Plot single vs double
plt.xscale('log')
plt.plot(narrayx,narray2,label = "Single Precision")
plt.plot(narrayx,narray, label = "Double Precision")
plt.title("Single vs Double precision", fontsize = 30)
plt.tick_params(axis='both', labelsize=15)
plt.xlabel(r"$\frac{r}{n}$",fontsize = 50)
plt.ylabel("Solution", fontsize = 25)

plt.legend()
plt.show()
