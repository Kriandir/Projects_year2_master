import numpy as np
import matplotlib.pyplot as plt
B = np.zeros((6,6))
B[0][2] = 1
B[1][0] = 1
B[1][3] = 1
B[1][4] = 1
B[2][1] = 1
B[2][5] = 1
B[3][0] = 1
B[3][2] = 1
B[4][3] = 1
B[5][1] = 1
B[5][2] = 1
B[5][4] = 1

# intialize matrix
def InitMatrix(B):

    A = np.zeros((6,6))
    for i in range(np.shape(B)[0]):
        normlist = []
        for j in range(np.shape(B)[0]):
            normlist.append(B[j][i])
        norm = sum(normlist)
        for j in range(np.shape(B)[0]):
            A[j][i] = B[j][i] / norm
    return A

# initialize random eigenvector
def InitVector():
    x = np.random.rand(6)
    z = np.sum(x)
    vectlist = []
    for i in x:
        vectlist.append(i/z)
    b = np.array(vectlist)
    return b

# Power iteration's to find the largest absolute eigenvalue
def PowerIterate(M,N,b):
    i = 0
    stoplist=[0]
    while i < N:
        # calculate  Mb
        b_1 = np.dot(M, b)
        # calculate the norm
        b_1_norm = np.linalg.norm(b_1)
        # re normalize the vector
        b = b_1 / b_1_norm

        lambdas = CalcLambda(b,M)
        # if no change in the residuals on order of 10^-4 return values
        residuals = M.dot(b) - lambdas * (b)
        if(np.linalg.norm(residuals,ord=2)<10**(-4)):
            return b,i,residuals,lambdas
        i+=1
    return b,i,residuals,lambdas

#  modifiy matrix for question 3.c
def Convect(M,alpha,size=6):
    e = np.ones(size)
    y = (1./size)*e
    P = alpha*M +((1-alpha)*y.dot(e.T))
    return P

# Find the eigenvalue using the RayleighQuotient:
def CalcLambda(powervect,A):
    return powervect.T.dot(A).dot(powervect)/((powervect.T).dot(powervect))

#  Init matrices
A = InitMatrix(B)
B[3][2] = 0
B[0][2] = 0
C = InitMatrix(B)
b = InitVector()

# print everything and call the functions
print "-------------A-----------------"
powervect,iters,residuals,lambdas = PowerIterate(A,2000,b)
node = np.argmax(abs(powervect)) +1
print "Lambda is: "+ str(lambdas)+" After "+ str(iters) + " iterations"
print "Eigenvector is: "+ str(powervect)
print "Highest node is: " + str(node)
print "Residual is: " +str(residuals)

# C is the matrix where the nodes from 3 to 4 and 3 to 1 are removed
print "--------------------------------"
print C
print "--------------------------------"
powervect,iters,residuals,lambdas = PowerIterate(C,2000,b)
node = np.argmax(abs(powervect)) +1
print "--------------C-----------------"
print "Lambda is: "+ str(lambdas)+" After "+ str(iters) + " iterations"
print "Eigenvector is: "+ str(powervect)
print "Highest node is: " + str(node)
print "Residual is: " +str(residuals)

# P is the modified C matrix here for alpha = 0.95
P = Convect(C,0.95)
powervect,iters,residuals,lambdas = PowerIterate(P,2000,b)
node = np.argmax(abs(powervect)) +1
print "---------------P for alpha = 0.95----------------"
print "Lambda is: "+ str(lambdas)+" After "+ str(iters) + " iterations"
print "Eigenvector is: "+ str(powervect)
print "Highest node is: " + str(node)
print "Residual is: " +str(residuals)

# P is the modified C matrix here for alpha = 0.75
P = Convect(C,0.75)
powervect,iters,residuals,lambdas = PowerIterate(P,2000,b)
node = np.argmax(abs(powervect)) +1
print "---------------P for alpha = 0.75----------------"
print "Lambda is: "+ str(lambdas)+" After "+ str(iters) + " iterations"
print "Eigenvector is: "+ str(powervect)
print "Highest node is: " + str(node)
print "Residual is: " +str(residuals)

list95 = []
list75 = []
# compare iterations with an alpha of 0.95 and 0.75 over 1000 times
for i in range(1000):
    b = InitVector()
    P = Convect(C,0.95)
    powervect,iters,residuals,lambdas = PowerIterate(P,2000,b)
    list95.append(iters)
    P = Convect(C,0.75)
    powervect,iters,residuals,lambdas = PowerIterate(P,2000,b)
    list75.append(iters)

# plot iteration comparison
plt.plot(list95, label= r"$\alpha$ = 0.95")
plt.plot(list75, label= r"$\alpha$ = 0.75")
plt.xlabel("FunctionCall",fontsize=20)
plt.ylabel("Number of Iterations",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Excercise 3, iteration comparison",fontsize=30)
plt.show()


# Perform matrix modification for alpha from 0 to 1 in steps of 0.01
alphalist = []
a = np.arange(0,1,0.01)
for i in a:
    P = Convect(C,i)
    powervect,iters,residuals,lambdas = PowerIterate(P,2000,b)
    alphalist.append(iters)
plt.scatter(a,alphalist)
plt.xlabel(r"$\alpha$",fontsize=20)
plt.ylabel("Number of Iterations",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Excercise 3 different Alpha's",fontsize=30)
plt.show()
