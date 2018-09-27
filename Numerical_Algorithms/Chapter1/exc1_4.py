import math
import numpy as np
import matplotlib.pyplot as plt

steps = 20
ylist = []


for i in range(steps):
    ylist.append((1. + 1./10**i)**(10**i))
errorlist = abs(np.array(ylist)-np.exp(1))

print ylist[15]
print errorlist

plt.plot(ylist)
plt.show()
plt.plot(errorlist)
plt.show()
