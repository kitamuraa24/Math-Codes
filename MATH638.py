import numpy as np
import Riemann_Solver as RS

def linear_u0(N):
    uL = -1 * np.ones(int(N/2))
    uR = np.ones(int(N/2))
    u0 = np.append(uL, uR) 
    return u0

def burgers_u0(N, x):
    u0 = np.zeros(N)
    idx = int(N/4)
    for i in range(idx, 2*idx):
        u0[i] = 1 + x[i]
    for i in range(2*idx, 3*idx):
        u0[i] = 1- x[i]
    return u0

def buckleyLeverett_u0(N):
    u0 = np.ones(N)
    u0[int(N/2):] = 0
    return u0

if __name__=="__main__":
    N = 200
    x = np.linspace(-1.99, 1.99, N)
    CFL = 0.3
    t_end = [0.5, 2, 20]
    u0 = linear_u0(N)
    xb = [-2, 2]
    #u0 = burgers_u0(N, x)
    #u0 = buckleyLeverett_u0(N)
