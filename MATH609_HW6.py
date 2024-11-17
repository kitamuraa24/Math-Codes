import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import Jacobi_GaussSeidel as JGS

def buildA_b(h):
    """
    Algorithm corresponds to FDM 2nd order, with Dirichlet BC
    """
    n = int(1/h - 1)  # Number of interior points per axis
    N = n ** 2
    A = np.zeros((N, N))
    # Build B matrix
    B = np.zeros((n,n))
    np.fill_diagonal(B, -4/h**2)
    for i in range(n-1):
        B[i, i+1] = 1/h**2
        B[i+1, i] = 1/h**2
    # Fill the blocks
    for i in range(n):
        # Set the B block on the diagonal
        A[i * n: (i + 1) * n, i * n: (i + 1) * n] = B
        
        # Set the I block on the subdiagonal and superdiagonal
        if i < n - 1:
            A[(i + 1) * n: (i + 2) * n, i * n: (i + 1) * n] = 1/h**2 * np.eye(n)  # subdiagonal
            A[i * n: (i + 1) * n, (i + 1) * n: (i + 2) * n] = 1/h**2 * np.eye(n)  # superdiagonal
    
    b_tile = np.zeros(n)
    b_tile[-1] = -1/h**2
    b = np.tile(b_tile, n)
    x = np.arange(0-h/2, 1+3*h/2, h)  
    y = np.arange(0-h/2, 1+3*h/2, h)  
   
    return A, b, x, y

def add_bc(n, W):
    x0 = np.tile(0, n)
    x1 = np.tile(0, n)
    y0 = np.tile(0, n+2)
    y1 = np.tile(1, n+2)
    W = np.insert(W, 0, x0, axis=1)
    W = np.insert(W, n+1, x1, axis=1)
    W = np.insert(W, 0, y0, axis=0)
    W = np.insert(W, n+1, y1, axis=0)
    return W

if __name__ == "__main__":
    h_list = [1/4, 1/8, 1/16]
    for h in h_list:
        n = int(1/h - 1)
        xtol = 1e-2
        A, b, x, y = buildA_b(h)
        rho_Bj = 1 - 0.5 * np.pi**2 * h**2
        omega = 2/(1 + np.sqrt(1 - rho_Bj**2))
        Solver = JGS.Solver(A, b, 25, xtol, w=omega)
        method_list = {"J": Solver.Jacobi(),
                    "GS": Solver.Gauss_Seidel(),
                    "SoR": Solver.SoR()}
        # method_list = {"J": Solver.Jacobi()}
        for m in method_list:
        # Iterative Method to solve
            w = method_list[m]
            X, Y = np.meshgrid(x, y)      # Create the 2D grid
            W = w.reshape((n, n), order='F') # iterates j first, then i
            W = add_bc(n, W)
            plt.figure(figsize=(8, 6))
            pc = plt.pcolormesh(X, Y, W, cmap=cm.coolwarm, shading='flat')  # Discrete cells
            plt.colorbar(pc, label="Temperature (u)")
            plt.title(f"Plot of Temperature Distribution (Method: {m})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
    # Segment of code to handle GD and CG
    for h in h_list:
        n = int(1/h - 1)
        xtol = 1e-2
        A, b, x, y = buildA_b(h)