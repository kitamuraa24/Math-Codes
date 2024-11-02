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
    n = len(f)

    # Forward Elimination
    for i in range(1, n):
        w = a[i-1] / b[i-1]
        b[i] = b[i] - w * q[i-1]
        f[i] = f[i] - w * f[i-1]

    # Backward Substitution
    x = np.zeros(n)
    x[-1] = f[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (f[i] - q[i] * x[i+1]) / b[i]

    return x