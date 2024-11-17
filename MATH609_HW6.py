import numpy as np
import matplotlib.pyplot as plt

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
    x = np.arange(0, 1+h, h)  
    y = np.arange(0, 1+h, h)  
   
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
    h = 1/4
    n = int(1/h - 1)
    xtol = 1e-2
    A, b, x, y = buildA_b(h)
    w = np.linalg.solve(A, b)
    X, Y = np.meshgrid(x, y)      # Create the 2D grid
    W = w.reshape((n, n))
    W = add_bc(n, W)
    print(W)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, W, levels=20, cmap='viridis')  # Filled contour plot
    plt.colorbar(contour, label="Temperature (Φ)")
    plt.title("Contour Plot of Temperature Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")  # Keep aspect ratio square
    plt.show()
    