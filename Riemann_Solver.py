import numpy as np
from scipy.optimize import minimize_scalar

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
        self.flux_speed = {"Linear": self.Linear_speed,
                           "Burgers": self.Burgers_speed,
                           "Buckley_Leverett": self.Buckley_Leverett_speed}
    def Linear(self, u):
        return u
    
    def Linear_speed(self, u):
        return -1
    
    def Burgers(self, u):
        return u**2/2
    
    def Burgers_speed(self, u):
        return -u

    def Buckley_Leverett(self, u):
        return u**2/(u**2 + (1-u)**2)
    
    def Buckley_Leverett_speed(self, u):
        return -(2*u-2*u**2)/(2*u**2-2*u+1)**2
    
    def minmod(self, a, b):
        if a >= 0 and b >= 0:
            return min(a, b)
        elif a <= 0 and b <= 0:
            return max(a, b)
        else:
            return 0
    
    def get_dt(self, h):
        speed = self.flux_speed.get(self.flux)
        if speed is None:
            raise ValueError("Flux not supported!")
        max_speed = minimize_scalar(speed, bounds=[0, 1])
        dt = self.CFL *h / (-max_speed.fun)
        return dt

    def LxF(self):
        print("Lax-Friedrich scheme is used\n")
        u_old = self.u0
        u_new = np.copy(u_old)
        h = self.h
        self.dt = self.get_dt(h)
        t, k = 0, 0
         # Select the appropriate flux function
        f = self.flux_types.get(self.flux)
        if f is None:
            raise ValueError("Flux not supported!")
        while t < self.t_end:
            dt = self.dt
            if t + dt > self.t_end:
                if k % 2 != 0:
                    dt = self.t_end - t
                else:
                    dt = 0
            t+=dt
            if k % 2 == 0:
                for i in range(len(self.u0)-2):
                    df = (f(u_old[i+1]) - f(u_old[i]))
                    du = (u_old[i+1] + u_old[i])
                    u_new[i] =  0.5 * du - dt/self.h * df
                u_new[-1] = u_new[0]
            else:
                for i in range(len(self.u0)-1, 0, -1):
                    df = (f(u_old[i]) - f(u_old[i-1]))
                    du = u_old[i-1] + u_old[i]
                    u_new[i] = 0.5 * du -  dt/self.h * df
                u_new[0] = u_new[-1]
            k+=1
            u_old = np.copy(u_new)
            if self.verbose == True:
                print(f"Time Iter No: {k} | dt: {dt:.6f} | time: {t:.6f}")
        return u_old
    
    def NT(self):
        print("Nessyahu-Tadmor Scheme is used\n")
        u_old = self.u0
        u_new = np.copy(u_old)
        t, k = 0, 0
        h = self.h
        self.dt = self.get_dt(h)
        f = self.flux_types.get(self.flux)
        if f is None:
            raise ValueError("Flux not supported!")
        while t < self.t_end:
            dt = self.dt
            if t + dt > self.t_end:
                if k % 2 != 0:
                    dt = self.t_end - t
                else: 
                    dt = 0
            t+=dt
            if k % 2 == 0: #even k
                for i in range(len(self.u0)):
                    if i == 0:
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[-1]
                        df_ip2 = f(u_old[i+2])- f(u_old[i+1])
                        df_ip1 = f(u_old[i+1]) - f(u_old[i]) 
                        df_i = f(u_old[i]) - f(u_old[-1])
                    elif i == len(self.u0)-2:
                        du_ip2 = u_old[0] - u_old[i+1]
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        df_ip2 = f(u_old[0])- f(u_old[i+1])
                        df_ip1 = f(u_old[i+1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                    elif i == len(self.u0) - 1:
                        du_ip2 = u_old[1] - u_old[0]
                        du_ip1 = u_old[0] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        df_ip2 = f(u_old[1])- f(u_old[0])
                        df_ip1 = f(u_old[0]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                    else:
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        df_ip2 = f(u_old[i+2]) - f(u_old[i+1])
                        df_ip1 = f(u_old[i+1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                    upi = self.minmod(du_ip1, du_i)
                    up_ip1 = self.minmod(du_ip2, du_ip1)
                    fp_i = self.minmod(df_ip1, df_i)
                    fp_ip1 = self.minmod(df_ip2, df_ip1)
                    u_i_half = u_old[i] - 0.5 * dt/h * fp_i
                    # upi, up_ip1, fp_i, fp_ip1 = 0, 0, 0, 0
                    u_i_half = u_old[i] - 0.5 * dt/h * fp_i
                    if i == len(self.u0) - 1:
                        u_ip1_half = u_old[0] -0.5 * dt/h *fp_ip1
                        u_new[i] = 0.5 * (u_old[i] + u_old[0]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))
                    else:
                        u_ip1_half = u_old[i+1] -0.5 * dt/h *fp_ip1
                        u_new[i] = 0.5 * (u_old[i] + u_old[i+1]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))                   
            else: #odd k
                for i in range(len(self.u0) -1, -1, -1):
                    if i == len(self.u0) - 1:
                        du_ip2 = u_old[0] - u_old[i]
                        du_ip1 = u_old[i] - u_old[i-1]
                        du_i = u_old[i-1] - u_old[i-2]
                        df_ip2 = f(u_old[0])- f(u_old[i])
                        df_ip1 = f(u_old[i]) - f(u_old[i-1])
                        df_i = f(u_old[i-1]) - f(u_old[i-2])
                    elif i == 0:
                        du_ip2 = u_old[i+1] - u_old[i]
                        du_ip1 = u_old[i] - u_old[-1]
                        du_i = u_old[-1] - u_old[-2]
                        df_ip2 = f(u_old[i+1])- f(u_old[i])
                        df_ip1 = f(u_old[i]) - f(u_old[-1]) 
                        df_i = f(u_old[-1]) - f(u_old[-2])
                    elif i == 1:
                        du_ip2 = u_old[i+1] - u_old[i]
                        du_ip1 = u_old[i] - u_old[i-1]
                        du_i = u_old[i-1] - u_old[-1]
                        df_ip2 = f(u_old[i+1])- f(u_old[i])
                        df_ip1 = f(u_old[i]) - f(u_old[i-1]) 
                        df_i = f(u_old[i-1]) - f(u_old[-1])
                    else:
                        du_ip2 = u_old[i+1] - u_old[i]
                        du_ip1 = u_old[i] - u_old[i-1]
                        du_i = u_old[i-1] - u_old[i-2]
                        df_ip2 = f(u_old[i+1]) - f(u_old[i])
                        df_ip1 = f(u_old[i]) - f(u_old[i-1])
                        df_i = f(u_old[i-1]) - f(u_old[i-2])
                    upi = self.minmod(du_ip1, du_i)
                    up_ip1 = self.minmod(du_ip2, du_ip1)
                    fp_i = self.minmod(df_ip1, df_i)
                    fp_ip1 = self.minmod(df_ip2, df_ip1)
                    # upi, up_ip1, fp_i, fp_ip1 = 0, 0, 0, 0
                    u_ip1_half = u_old[i] -0.5 * dt/h *fp_ip1
                    if i == 0:
                        u_i_half = u_old[-1] -0.5 * dt/h *fp_ip1
                        u_new[i] = 0.5 * (u_old[-1] + u_old[i]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))
                    else:
                        u_i_half = u_old[i-1] - 0.5 * dt/h * fp_i
                        u_new[i] = 0.5 * (u_old[i] + u_old[i-1]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))
            k+=1
            u_old = np.copy(u_new)
            if self.verbose == True:
                print(f"Time Iter No: {k} | dt: {dt:.6f} | time: {t:.6f}")
        return u_old