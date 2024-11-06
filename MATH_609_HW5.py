import numpy as np
import matplotlib.pyplot as plt
import Thomas_Alg as TA
import GD_CG as GDCG

def calc_Linf_norm(u, phi):
    L_inf = np.linalg.norm(u - phi, ord=np.inf)
    return L_inf

def analytical_soln(x):
    return np.exp(x**2)

# For Problem 3
def pre_TA3(a_i, q_i, n):
    if n - 1 > 0:
        a = np.repeat(a_i, n - 1)
        q = np.repeat(q_i, n - 1)
    else: a, q = None, None
    return a, q

def build_f3(a, b, n):
    if n == 1:
        f = b
    else:
        f = np.zeros(n)
        f[0] = a
        f[-1] = b
    return f

def build_A3(a_i, b, q_i, n):
    A = np.zeros((n, n))
    A[0, 0], A[0, 1] = b[0], q_i
    A[-1, -1], A[-1, -2] = b[-1], a_i
    for i in range(1, len(A)-1):
        A[i, i-1] = a_i
        A[i, i] = b[i]
        A[i, i+1] = q_i
    return A
    
def calc_b_i3(x, n, h):
    b = np.zeros(n)
    for i in range(n):
        b[i] = 2 + h**2*(4*x[i+ 1]**2 + 2)
    return b
#For Problem 4
def set_g(n):
    g = np.zeros(n+2)
    for i in range(n+2):
        g[i] = 4*x[i]**2 + 2
    return g

def set_a(n, g, h):
    a = np.zeros(n-1)
    for i in range(n-1):
        a[i] = g[i+1]/12 - 1/h**2
    return a

def set_b(n, g, h):
    b = np.zeros(n)
    for i in range(n):
        b[i] = 2/h**2 + 5/6 * g[i+1]
    return b

def set_q(n, g, h):
    q = np.zeros(n-1)
    for i in range(n-1):
        q[i] = g[i+2]/12 - 1/h**2
    return q

def set_f (n, g, h):
    f = np.zeros(n)
    f[0] = 1/h**2 - g[0]/12
    f[-1] = np.exp(1) * (1/h**2 - g[-1]/12)
    return f


if __name__ == '__main__':
    u_left, u_right = 1, np.exp(1)
    problem = 3
    if problem == 3:
        p_list = np.arange(1, 15, 1)
        max_iter, rtol = 50000, 1e-6
        methods = ["Thomas-Algorithm", "Gradient-Descent", "Conjugate-Gradient"]
        h_list = []
        L_inf_cg_list, L_inf_ta_list, L_inf_gd_list = [], [], []
        for p in p_list:
            h = 0.5**(p)
            x = np.arange(0, 1 + h, h)
            n = len(x) - 2
            b = calc_b_i3(x, n, h)
            a_i, q_i = -1, -1
            a, q = pre_TA3(a_i, q_i, n)
            f = build_f3(u_left, u_right, n)
            if p == 1:
                # Single forward solve
                u_f = (np.exp(1) + 1) / (h**2*(4*x[1]**2 + 2) + 2)
                u_f = np.insert(u_f, 0, u_left)
                u_f = np.insert(u_f, n, u_right)
                phi = analytical_soln(x)
                L_inf_f = calc_Linf_norm(u_f, phi)
                h_list.append(h)
            else:
                A = build_A3(a_i, b, q_i, n)
                # Solve with LU, GD, and CG
                u_ta = TA.thomas_alg(a, b, q, f)
                u0 = np.copy(u_ta) + 1e-2
                Sol = GDCG.Solver(u0, A, f, max_iter, rtol, False)
                u_gd = Sol.Gradient_Descent()
                u_cg = Sol.Conjugate_Gradient()
                u_ta = np.insert(u_ta, 0, u_left)
                u_ta = np.insert(u_ta, n+1, u_right)
                u_gd = np.insert(u_gd, 0, u_left)
                u_gd = np.insert(u_gd, n+1, u_right)
                u_cg = np.insert(u_cg, 0, u_left)
                u_cg = np.insert(u_cg, n+1, u_right)
                phi = analytical_soln(x)
                # Plotting
                plt.scatter(x, u_gd, s = 2, label=f"p={p}")
                #Post-Process
                L_inf_ta = calc_Linf_norm(u_ta, phi)
                L_inf_gd = calc_Linf_norm(u_gd, phi)
                L_inf_cg = calc_Linf_norm(u_cg, phi)
                h_list.append(h)
                L_inf_ta_list.append(L_inf_ta)
                L_inf_cg_list.append(L_inf_cg)
                L_inf_gd_list.append(L_inf_gd)
        # Tables
        for m in methods:
            i = 0
            if m == "Thomas-Algorithm":
                L_inf = L_inf_ta_list
            elif m == "Gradient-Descent":
                L_inf = L_inf_gd_list
            else:
                L_inf = L_inf_cg_list
            print(m)
            for h in h_list:
                if h == 1:
                    print(f"h: {h} | L_inf: {L_inf_f:.4e} | L_inf/h: {L_inf_f/h:.4e} | L_inf/h^2: {L_inf_f/h**2:.4e} | L_inf/h^3: {L_inf_f/h**3:.4e}")
                else:
                    print(f"h: {h} | L_inf: {L_inf[i]:.4e} | L_inf/h: {L_inf[i]/h:.4e} | L_inf/h^2: {L_inf[i]/h**2:.4e} | L_inf/h^3: {L_inf[i]/h**3:.4e}")
                    i+=1
        plt.plot(x, phi, label="Analytical solution")
        plt.legend()
        plt.grid()
        plt.show()
    elif problem == 4:
        p_list = np.arange(1, 7, 1)
        L_inf_list = []
        h_list = []
        for p in p_list:
            h = (0.5)**(p)
            h_list.append(h)
            x = np.arange(0, 1 + h, h)
            n = len(x) - 2
            g = set_g(n)
            a = set_a(n, g, h)
            b = set_b(n, g, h)
            q = set_q(n, g, h)
            f = set_f(n, g, h)
            if p == 1:
                u = np.array([(1/h**2-g[0]/12 + np.exp(1)*(1/h**2 - g[-1]/12))/(2/h**2 + 5/6*g[1])])
            else:
                u = TA.thomas_alg(a, b, q, f)
            u = np.insert(u, 0, u_left)
            u = np.insert(u, n+1, u_right)
            phi = analytical_soln(x)
            L_inf = calc_Linf_norm(u, phi)
            L_inf_list.append(L_inf)
            plt.plot(x, u, label=f"h = {h}")
            print(f"h: {h} | L_inf: {L_inf:.4e} | L_inf/h^3: {L_inf/h**3:.4e} | L_inf/h^4: {L_inf/h**4:.4e} | L_inf/h^5: {L_inf/h**5:.4e}")
        plt.plot(x, phi, label="Analytical Solution")
        plt.xlabel('x')
        plt.ylabel('$\phi$')
        plt.legend()
        plt.show()
        plt.loglog(h_list, L_inf_list)
        plt.grid(True)
        plt.title("Log-log plot of the error norm.")
        plt.show()


        