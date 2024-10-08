import numpy as np
import Jacobi_GaussSeidel as JGS

class Solver:
    def __init__(self, x0, A, b, max_iter, rtol, verbose=False):
        self.x0 = x0
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.rtol = rtol
        self.verbose = verbose

    def Gradient_Descent(self):
        x0 = self.x0
        r = self.b - np.dot(self.A, x0)
        for k in range(self.max_iter):
            alpha = np.dot(r, r) / (r.T @ self.A @ r)
            x1 = x0 + alpha * r
            r = self.b - self.A @ x1
            conv = np.linalg.norm(r)
            if conv < self.rtol:
                print(conv)
                break
            else:
                x0 = x1
        return x0
    
    def Conjugate_Gradient(self):
        x0 = self.x0
        r = self.b - np.dot(A, x0)
        d = r
        for k in range(self.max_iter):
            z = np.dot(A, d)
            alpha = np.dot(r, d) / np.dot(d, z)
            x1 = x0 + alpha * d
            r = r - alpha * z
            conv = np.linalg.norm(r)
            if conv < self.rtol:
                print(conv)
                break
            else:
                beta = np.dot(r, z) / np.dot(d, z)
                d = r + beta * d
                x0 = x1
        return x0

if __name__ == '__main__':
    A = np.array([[10, 1, 2, 3, 4],
                  [1, 9, -1, 2, -3],
                  [2, -1, 7, 3, -5],
                  [3, 2, 3, 12, -1],
                  [4, -3, -5, -1, 15]])
    b = np.array([12, -27, 14, -17, 12])
    x0 = np.array([0, 0, 0, 0, 0])
    max_iter, rtol = 100, 1e-8
    Sol = Solver(x0, A, b, max_iter, rtol)
    x_sol = Sol.Gradient_Descent()
    print(x_sol)
    x_sol = Sol.Conjugate_Gradient()
    print(x_sol)     
