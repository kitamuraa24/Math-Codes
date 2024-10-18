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
        

    def Linear(self, u):
        return u
    
    def Burgers(self, u):
        return u**2/2
    
    def Buckley_Leverett(self, u):
        return u**2/(u**2 + (1-u)**2)
    
    def minmod(self, a, b):
        if a >= 0 and b >= 0:
            return min(a, b)
        elif a <= 0 and b <= 0:
            return max(a, b)
        elif a*b < 0:
            return 0

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
                if k % 2 != 0:
                    dt = self.t_end - t
                else:
                    dt = 0
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
    
    def NT(self):
        #TO DO: Fix the Left sweeping (odd k)
        u_old = self.u0
        u_new = np.copy(u_old)
        t, k = 0, 0
        h = self.h
        try:
            f = self.flux_types[self.flux]
        except:
            "Flux not supported!"
        while t < self.t_end:
            u_max = np.max(self.u0)
            dt = self.CFL * self.h / u_max
            if t + dt > self.t_end:
                if k % 2 != 0:
                    dt = self.t_end - t
                else: dt = 0
            t+=dt
            if k % 2 == 0: #even k
                for i in range(len(self.u0)):
                    if i == 0:
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[-1]
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        df_ip1 = f(u_old[i+1]) - f(u_old[i]) 
                        df_i = f(u_old[i]) - f(u_old[-1])
                        df_ip2 = f(u_old[i+2])- f(u_old[i+1])
                    elif i == len(self.u0)-2:
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[0] - u_old[i+1]
                        df_ip1 = f(u_old[i+1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[0])- f(u_old[i+1])
                    elif i == len(self.u0) - 1:
                        du_ip1 = u_old[0] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[1] - u_old[0]
                        df_ip1 = f(u_old[0]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[1])- f(u_old[0])
                    else:
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        df_ip1 = f(u_old[i+1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[i+2]) - f(u_old[i+1])
                    upi = self.minmod(du_ip1, du_i)
                    up_ip1 = self.minmod(du_ip2, du_ip1)
                    fp_i = self.minmod(df_ip1, df_i)
                    fp_ip1 = self.minmod(df_ip2, df_ip1)
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
                for i in range(len(self.u0) -1, 0, -1):
                    if i == len(self.u0) - 1:
                        du_ip1 = u_old[0] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[1] - u_old[0]
                        df_ip1 = f(u_old[0]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[1])- f(u_old[0])
                    elif i == len(self.u0) - 2:
                        du_ip1 = u_old[-1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[0] - u_old[-1]
                        df_ip1 = f(u_old[-1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[0])- f(u_old[-1])
                    elif i == 0:
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[-1]
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        df_ip1 = f(u_old[i+1]) - f(u_old[i]) 
                        df_i = f(u_old[i]) - f(u_old[-1])
                        df_ip2 = f(u_old[i+2])- f(u_old[i+1])
                    else:
                        du_ip1 = u_old[i+1] - u_old[i]
                        du_i = u_old[i] - u_old[i-1]
                        du_ip2 = u_old[i+2] - u_old[i+1]
                        df_ip1 = f(u_old[i+1]) - f(u_old[i])
                        df_i = f(u_old[i]) - f(u_old[i-1])
                        df_ip2 = f(u_old[i+2]) - f(u_old[i+1])
                    upi = self.minmod(du_ip1, du_i)
                    up_ip1 = self.minmod(du_ip2, du_ip1)
                    fp_i = self.minmod(df_ip1, df_i)
                    fp_ip1 = self.minmod(df_ip2, df_ip1)
                    u_i_half = u_old[i] - 0.5 * dt/h * fp_i
                    if i == len(self.u0) -1:
                        u_ip1_half = u_old[0] -0.5 * dt/h *fp_ip1
                        u_new[i] = 0.5 * (u_old[i] + u_old[0]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))
                    else:
                        u_ip1_half = u_old[i+1] -0.5 * dt/h *fp_ip1
                        u_new[i] = 0.5 * (u_old[i] + u_old[i+1]) + 0.125 * (upi - up_ip1) \
                                - dt/h *(f(u_ip1_half) - f(u_i_half))
            k+=1
            u_old = np.copy(u_new)
            if self.verbose == True:
                print(f"Time Iter No: {k} | dt: {dt:.3f} | time: {t:.3f}")
        return u_old