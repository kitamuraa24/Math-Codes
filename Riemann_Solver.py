import numpy as np

class Riemann_Solver:
    def __init__(self, params, verbose=False):
        self.u0 = params[0]
        self.u = np.copy(self.u0)
        self.t_end = params[1]
        self.CFL = params[2]
        self.h = params[3]
        self.xbounds = params[4]
        self.flux = params[5]
        self.verbose = verbose
        self.flux_types = {"Linear": self.Linear,
                           "Burgers": self.Burgers,
                           "Buckley_Leverett": self.Buckley_Leverett}

    def Linear(u):
        return u
    
    def Burgers(u):
        return u**2/2
    
    def Buckley_Leverett(u):
        return u**2/(u**2 + (1-u)**2)
    
    def minmod(a, b):
        if a > 0 and b > 0:
            return a

    def LxF(self):
        u_old = self.u0
        u_new = np.copy(u_old)
        t, k = 0, 0
        try:
            f = self.flux_types[self.flux]
        except:
            "Flux not supported!"
        while t < self.t_end:
            u_max = np.max(self.u0)
            dt = self.CFL * self.h / u_max
            if t + dt > self.t_end:
                dt = self.t_end - t
            t+=dt
            if k % 2 == 0:
                for i in range(len(self.u0)-1):
                    u_new[i] = 0.5 * (u_old[i] + u_old[i+1]) 
                    - dt/self.h * (f(u_old[i+1]) -f(u_old[i]))
                u_new[-1] = u_new[0]
            else:
                for i in range(len(self.u0), 1):
                    u_new[i] = 0.5 * (u_old[i-1] + u_old[i])
                    - dt/self.h * (f(u_old[i]) - f(u_old[i-1]))
                u_new[0] = u_new[-1]
            k+=1
            u_old = np.copy(u_new)
            if self.verbose == True:
                print(f"Time Iter No: {k} | dt: {dt:.3f} | time: {t:.3f}")
        return u_old
            