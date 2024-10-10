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
        print('Gradient Descent is used')
        x0 = self.x0
        r = self.b - np.dot(self.A, x0)
        for k in range(self.max_iter):
            alpha = np.dot(r, r) / (r.T @ self.A @ r)
            x1 = x0 + alpha * r
            r = self.b - self.A @ x1
            conv = np.linalg.norm(r)
            if self.verbose == True:
                print(f"Iteration {k}: r= {conv:.5f}, x= {x1}")
            if conv < self.rtol:
                print(f"Total Iterations: {k}")
                x0 = x1
                break
            elif k == self.max_iter-1:
                print("Max Iterations Reached.")
            else:
                x0 = x1
        return x0
    
    def Conjugate_Gradient(self):
        print('Conjugate Gradient is used.')
        x0 = self.x0
        r = self.b - np.dot(A, x0)
        d = -r
        for k in range(self.max_iter):
            z = np.dot(A, d)
            alpha = np.dot(r, d) / np.dot(d, z)
            x1 = x0 + alpha * d
            r = r - alpha * z
            conv = np.linalg.norm(r)
            if self.verbose == True:
                print(f"Iteration {k}: r= {conv}, x= {x1}")
            if conv < self.rtol:
                x0 = x1
                print(f"Total Iterations: {k}")
                break
            elif k == self.max_iter-1:
                print("Max Iterations Reached.")
            else:
                beta = np.dot(r, z) / np.dot(d, z)
                d = -r + beta * d
                x0 = x1
        return x0

def build_A(n):
    A = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (1 + i + j)
    return A

def build_b(A):
    n = len(A)
    b = np.zeros(n)
    for i in range(n):
        b[i] = 1/3 * np.sum(A[i, :])
    return b

if __name__ == '__main__':
    # N = [16, 32]
    # for n in N:
    #     A = build_A(n)
    #     b = build_b(A)
    #     x0 = np.zeros(n)
    #     Sol = Solver(x0, A, b, 7000, 1e-6, False)
    #     x_sol = Sol.Gradient_Descent()
    #     print(f"For {n}, the solution using Gradient Descent is: {x_sol}\n")
    #     x_sol = Sol.Conjugate_Gradient()
    #     print(f"For {n}, the solution using Conjugate Gradient is: {x_sol}\n")   

    A = np.array([[10, 1, 2, 3, 4],
                  [1, 9, -1, 2, -3],
                  [2, -1, 7, 3, -5],
                  [3, 2, 3, 12, -1],
                  [4, -3, -5, -1, 15]])
    b = np.array([12, -27, 14, -17, 12])
    x0 = np.array([0, 0, 0, 0, 0])
    max_iter, rtol = 100, 1e-6
    Sol = Solver(x0, A, b, max_iter, rtol, True)
    x_sol = Sol.Gradient_Descent()
    print(f"The solution using Gradient Descent is: {x_sol}\n")
    x_sol = Sol.Conjugate_Gradient()
    print(f"The solution using Conjugate Gradient is: {x_sol}")     
    Sol = JGS.Solver(A, b, max_iter, rtol, False)
    x_sol = Sol.Jacobi()
    print(f"The solution using Jacobi Method is: {x_sol}")
    x_sol = Sol.Gauss_Seidel()
    print(f"The solution using Gauss Seidel is: {x_sol}")