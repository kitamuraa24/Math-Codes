import numpy as np
import matplotlib.pyplot as plt
import Riemann_Solver as RS

plt.rcParams["font.size"] = "14"
plt.rcParams["figure.figsize"] = [8,6]
plt.rcParams["figure.dpi"] = '300'

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

def trial(N):
    u0 = np.zeros(N)
    u0[int(N/4):int(N/2)] = 1
    return u0

if __name__=="__main__":
    N = 200
    h = 4/N
    CFL = 0.3
    t_end = [0.5, 2, 20]
    xb = [-2, 2]
    x = np.linspace(xb[0] +h/2, xb[1] - h/2, N)
    flux_types = {'Linear': linear_u0(N), 
                  'Burgers': burgers_u0(N, x),
                'Buckley_Leverett': buckleyLeverett_u0(N)}
    for flux in flux_types:
        try:
            u0 = flux_types[flux]
        except:
            "Flux type not supported"
        for t in t_end:
            params = [u0, t, CFL, h, xb, flux]
            Solver = RS.Riemann_Solver(params, verbose=True)
            u1 = Solver.LxF()
            u2 = Solver.NT()
            plt.figure(1)
            plt.title(f"{flux} at time {t} s")
            plt.grid(True)
            plt.ylabel("u")
            plt.xlabel("x")
            plt.plot(x, u1, label = 'LxF')
            plt.plot(x, u2, label='NT')
            plt.legend()
            plt.show()
    # u0 = trial(N)
    # flux = "Linear"
    # t = 1
    # params = [u0, t, CFL, h, xb, flux]
    # Solver = RS.Riemann_Solver(params, verbose=True)
    # u1 = Solver.LxF()
    # u2 = Solver.NT()
    # plt.figure(1)
    # plt.title(f"{flux} at time {t} s")
    # plt.grid(True)
    # plt.ylabel("u")
    # plt.xlabel("x")
    # plt.plot(x, u0, label='Initial')
    # plt.plot(x, u1, label = 'LxF')
    # plt.plot(x, u2, label='NT')
    # plt.legend()
    # plt.show()   
