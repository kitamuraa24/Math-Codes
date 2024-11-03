import numpy as np

def thomas_alg(a, b, q, f):
    """
    Solves Ax=b for tridiagonal matrix A.

    Parameters:
    - a (array) : lower, subdiagonal (len n-1)
    - b (array) : main diagonal (len n)
    - q (array) : upper, subdiagonal (len n-1)
    - f (array) : solution vector (len n)

    Returns:
    - x (array) : solution vector to Ax = b (len n)
    """
    n = len(b)
    l = np.zeros(n-1)
    u = np.zeros(n)

    u[0] = b[0]
    for i in range(1, n):
        l[i-1] = a[i-1] / u[i-1]
        u[i] = b[i] - l[i-1] * q[i-1]
    
    # Step 1: Forward substitution to solve Ly = d
    y = np.zeros(n)
    y[0] = f[0]
    for i in range(1, n):
        y[i] = f[i] - l[i - 1] * y[i - 1]

    # Step 2: Back substitution to solve Ux = y
    x = np.zeros(n)
    x[-1] = y[-1] / u[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - q[i] * x[i + 1]) / u[i]

    return x