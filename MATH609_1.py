import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '14'
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = '300'

EPS = 1

def thomas_alg(n):
    xlen = 2**n - 1
    h = 2**(-n)
    a = -EPS
    b = 2*EPS + h**2
    if n == 1:
        p = h
        f = 2*h + 1
        x = f/b * h**2
    else:
        
        #Construct u, l, q
        u = np.zeros(xlen)
        l = np.zeros(xlen-1)
        q = np.repeat(-EPS, xlen-1)
        u[0] = b
        l[0] = a/u[0]
        for i in range(1, xlen):
            u[i] = b - l[i-1]*q[i-1]
            if i == xlen -1:
                continue
            else:
                l[i] = a/u[i-1]
        u *= 1/h**2
        q *= 1/h**2
        # Construct f:
        f = np.zeros(xlen)
        p = np.zeros(xlen)
        for i in range(xlen):
            p[i] = i*h # This is supposed to be x_i that gets plotted
            f[i] = 2*p[i] + 1
        
        # Forward Elim:
        y = np.zeros(xlen)
        y[0] = f[0]
        for i in range(1, xlen):
            y[i] = f[i] - l[i-1]*y[i-1] 

        # Backward Sub:
        x = np.zeros(xlen)
        x[-1] = y[-1]/u[-1]
        for i in range(xlen-2, -1, -1):
            x[i] = (y[i] - x[i+1]*q[i]) / u[i]
        
    return p, x

# def direct_sol(n):
#     xlen = 2**n-1
#     h = 2**(-n)
#     A = np.zeros(shape=(xlen, xlen))
#     A[0,0], A[0, 1] = 2*EPS + h**2, -EPS
#     for i in range(1, xlen-1):
#         A[i, i-1] = -EPS
#         A[i, i] = 2*EPS + h**2
#         A[i, i+1] = -EPS
#     A[-1, -1], A[-1, -2] = 2*EPS + h**2, -EPS
#     A *= 1/h**2
#     f = np.zeros(xlen)
#     for i in range(xlen):
#         f[i] = 2*i*h + 1
#     x = np.linalg.solve(A, f)
#     return x
    



    