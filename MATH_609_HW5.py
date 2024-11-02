import numpy as np
import Thomas_Alg as TA
import GD_CG as GDCG

def pre_TA(a_i, b_i, q_i, n):
    n = n - 2
    b = np.repeat(b_i, n)
    if n - 1 > 0:
        a = np.repeat(a_i, n -1)
        q = np.repeat(q_i, n - 1)
    else: a, q = None, None
    return a, b, q

def build_f(a, b, n):
    if n == 3:
        f = b
    else:
        f = np.zeros(n-2)
        f[0] = a
        f[-1] = b
        for i in range(1, n-1):
            f[i] = 0 
    return f

def buildA(a_i, b_i, q_i, n):
    A = np.array(shape=(n-2, n-2))
    

if __name__ == '__main__':
    p_list = np.arange(1, 15, 1)
    for p in p_list:
        h = 0.5**(p)
        x = np.arange(0, 1 + h, h)
        n = len(x)

        