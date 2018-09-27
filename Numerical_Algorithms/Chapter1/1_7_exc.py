import math
import numpy as np
import matplotlib.pyplot as plt

steps = 60
sinlist1 = []
sinlist2= []
hlist = []

def Derrivative(x,h):
    y = (fx((x+h)) - fx(x))/(h)
    return y
def Centered(x,h):
    y = (fx((x+h)) - fx(x-h))/(2*h)
    return y

def fx(x):
    return np.sin(x)


for i in np.arange(1,steps,1):
    x=1.
    h = 1./(2**i)
    hlist.append(h)
    sinlist1.append(Derrivative(x,h))
    sinlist2.append(Centered(x,h))

errorlist1 = abs(np.array(sinlist1)-(np.cos(1.)))
errorlist2 = abs(np.array(sinlist2)-(np.cos(1.)))
# print sinlist
print errorlist1
plt.plot(np.log(hlist),sinlist1)
plt.plot(np.log(hlist),sinlist2)
plt.show()
# plt.plot(np.log(hlist),sinlist2)
plt.plot(np.log(hlist),np.log(errorlist1))
plt.plot(np.log(hlist),np.log(errorlist2))
plt.show()
