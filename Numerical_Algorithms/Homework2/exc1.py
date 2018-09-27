# EXCERCISE 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Generate vector with random entries
x = np.random.rand(100)
# set random entries in the vector to 0
for i in np.nditer(x,op_flags=["readwrite"]):
    if (np.random.randint(4)) == 1:
        i[...] = 0
    else:
        continue
# define pnorms
pnorm = [1,2,3,10,100,np.inf]
pnormvalues = []
# apply pnorms to the vector using np.linalg.norm
for i in range(len(pnorm)):

    pnormvalues.append(np.linalg.norm(x,ord=pnorm[i]))

print '%0.4f & %0.4f & %0.4f & %0.4f & %0.4f & %0.4f' %(pnormvalues[0],pnormvalues[1],pnormvalues[2],pnormvalues[3],pnormvalues[4],pnormvalues[5])

# Set new pnorm values
pnormvaluesp = []
pnormp = [0.5,0.1,0.01,0.001]
# apply pnorms to vector
for i in range(len(pnormp)):

    pnormvaluesp.append(sum(x**pnormp[i]))

print '%0.4f & %0.4f & %0.4f & %0.4f ' %(pnormvaluesp[0],pnormvaluesp[1],pnormvaluesp[2],pnormvaluesp[3])

# find out how many entries in vector x are non zero
indicies = np.nonzero(x)
print np.shape(indicies)[1]
