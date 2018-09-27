    h = -1
    print U
    for i in np.arange(-1,-np.shape(U)[0]-1,-1):
        for k in np.arange((i-1),-np.shape(U)[0]-1,-1):
            d = 0

            if U[k][h] == 0:
                continue
            else:
                for g in np.arange(-1,-np.shape(U)[1]-1,-1):
                    if U[k][g] == 0:
                        continue

                    if U[k][g] != 0:
                        if d == 0:
                            d = U[k][i]/U[i][i]
                            I[k][i] -= d
                        U[k][g] -= d * U[i][g]
                        I[k][g] -= d * I[i][g]
        h-=1
