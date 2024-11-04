import numpy as np
import matplotlib.pyplot as plt
import Thomas_Alg as TA
import GD_CG as GDCG

def pre_TA(a_i, q_i, n):
    if n - 1 > 0:
        a = np.repeat(a_i, n -1)
        q = np.repeat(q_i, n - 1)
    else: a, q = None, None
    return a, q

def build_f(a, b, n, h):
    if n == 1:
        f = b
    else:
        f = np.zeros(n)
        f[0] = a
        f[-1] = b
    return f

def build_A(a_i, b, q_i, n):
    A = np.zeros((n, n))
    A[0, 0], A[0, 1] = b[0], q_i
    A[-1, -1], A[-1, -2] = b[-1], a_i
    for i in range(1, len(A)-1):
        A[i, i-1] = a_i
        A[i, i] = b[i]
        A[i, i+1] = q_i
    return A
    
def calc_b_i(x, n, h):
    b = np.zeros(n)
    for i in range(n):
        b[i] = 2 + h**2*(4*x[i+ 1]**2 + 2)
    return b

def find_Linf_norm(u, phi):
    L_inf = np.linalg.norm(u - phi, ord=np.inf)
    return L_inf

def analytical_soln(x):
    return np.exp(x**2)

if __name__ == '__main__':
    p_list = np.arange(2, 15, 1)
    u_left, u_right = 1, np.exp(1)
    max_iter, rtol = 1000000000000000, 1e-4
    for p in p_list:
        h = 0.5**(p)
        x = np.arange(0, 1 + h, h)
        n = len(x) - 2
        b = calc_b_i(x, n, h)
        a_i, q_i = -1, -1
        a, q = pre_TA(a_i, q_i, n)
        f = build_f(u_left, u_right, n, h)
        A = build_A(a_i, b, q_i, n)
        # Solve with LU, GD, and CG
        u_ta = TA.thomas_alg(a, b, q, f)
        u0 = np.zeros(n)
        Sol = GDCG.Solver(u0, A, f, max_iter, rtol)
        u_gd = Sol.Gradient_Descent()
        u_cg = Sol.Conjugate_Gradient()
        u_ta = np.insert(u_ta, 0, u_left)
        u_ta = np.insert(u_ta, n, u_right)
        u_gd = np.insert(u_gd, 0, u_left)
        u_gd = np.insert(u_gd, n, u_right)
        u_cg = np.insert(u_cg, 0, u_left)
        u_cg = np.insert(u_cg, n, u_right)
        phi = analytical_soln(x)
        # Plotting
        plt.scatter(x, u_gd, s = 2, label=f"p={p}")
        #Post-Process
        L_inf_ta = find_Linf_norm(u_ta, phi)
        L_inf_gd = find_Linf_norm(u_gd, phi)
        L_inf_cg = find_Linf_norm(u_cg, phi)
        print(f"p = {p}")
        print(f"L_inf TA: {L_inf_ta:.4e}")
        print(f"L_inf GD: {L_inf_gd:.4e}")
        print(f"L_inf CG: {L_inf_cg:.4e}")
    plt.plot(x, phi, label="Analytical solution")
    plt.legend()
    plt.grid()
    plt.show()



        