import numpy as np
import matplotlib.pyplot as plt

# initialize tridiagonal matrix A
A = np.zeros((150,150))
for i in range(np.shape(A)[0]):
    for j in range(np.shape(A)[0]):
        if i == j:
            A[i][j] = -2
        if i == j+1:
            A[i][j] = 1
        if i == j-1:
            A[i][j] = 1


# Perform QR factorization
def QRfactorization(A,N,i=0):
    Q,R = np.linalg.qr(A)
    A = R.dot(Q)
    i+=1
    if(i<N):
        return QRfactorization(A,N,i)
    else:
        # compute diagonal and sort the list (something weird happens where the last eigenvalues are swapped in position)
        x = np.array(np.diagonal(A))
        return np.sort(x)

# plot results and sort the theoretical values because of the reason given aboveself.
linalgfunc = np.sort(np.linalg.eig(A)[0])
QR10 = QRfactorization(A,10)
QR100 = QRfactorization(A,100)
QR500 = QRfactorization(A,500)
x = np.arange(0,np.shape(A)[0],1)
plt.scatter(x,QR10,color ="r",s=12, label="10 QR iterations")
plt.scatter(x,QR100, color = "g",s=12, label="100 QR iterations")
plt.scatter(x,QR500, color = "b",s=12, label="500 QR iterations")
plt.plot(x,linalgfunc,color = "k",label = "np.linalg.eig")
plt.xlabel("Eigenvalue no.",fontsize=20)
plt.ylabel("Eigenvalues",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Excercise 4 eigenvalue comparison",fontsize=30)
plt.show()

# compute the error compared to the np.linalg.eig function and plot them
error10 = abs(linalgfunc-QR10)
error100 = abs(linalgfunc-QR100)
error500 = abs(linalgfunc-QR500)
plt.scatter(x,error10,color ="r",s=12, label="error on QR 10")
plt.scatter(x,error100, color = "g",s=12, label="error on QR 100")
plt.scatter(x,error500, color = "b",s=12, label="error on QR 500")
plt.axhline(0,color = "k",label = "zero line")
plt.xlabel("Eigenvalue no.",fontsize=20)
plt.ylabel("Absolute eigenvalue error",fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.legend()
plt.title("Excercise 4 error of QR compare to np.linalg.eig",fontsize=30)
plt.show()
