import numpy as np
import math


# Function returning F(x)
def Fx(x):
    return(np.tan(x))
# Function returning hat F(x)
def AFx(x):
    a = (x-(1./6)*(x**3))
    b = (1-((1./2)*(x**2)))
    return(a/b)

# Calc Forward error
def Forward(x):
    return(AFx(x)-Fx(x))

# Calc Backward error
def Backward(x):
    return(np.arctan(AFx(x))-x)

# Function returning derivative hat F(x)
def dAFx(x):
    return((4 + 0.333333*x**4)/(2 - x**2)**2)

# Function returning Relative Condition number evaluated around F(x)
def dRelCond(x):
    return((x * np.square(1/(np.cos(x)))/Fx(x)))

# Function returning Relative Condition number evaluated around hat F(x)
def adRelCond(x):
    return((x * dAFx(x))/AFx(x))

# Print function
def Printfunction(x):
    print("For x = %0.2f" % x)
    print("----------------------------------")
    print('Forward: %0.2e ' %(Forward(x)))
    print('Backward: %0.2e ' %(Backward(x)))
    print('Relative condition evaluating f(x): %0.2e' %(dRelCond(x)))
    print('Relative condition evaluating af(x): %0.2e' %(adRelCond(x)))
    print("----------------------------------")
    return 0

# Initializing values and print functions
fx = Fx(1)
afx = AFx(1)
print(fx)
print(afx)
Printfunction(1)
Printfunction(1.4)
