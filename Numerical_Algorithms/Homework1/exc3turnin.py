import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit


def Frloop (n,a,r):
    return(a*((1.+(r/n))))

def Fr(n,a,r):
    return(a*np.exp(n*np.log(1.+(r/n))))

r = 0.05
nsteps = [1,4,12,365]
sollist = []
astepstotal = []

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

exactsolution = []
for i in nsteps:
    a = 100
    exactsolution.append(Fr(i,a,r))

for i in range(len(exactsolution)):
    print "For an amount of steps of %0.f we get a final compound interest of: %0.5f" %(nsteps[i],exactsolution[i])
