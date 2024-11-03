import numpy as np

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
        r = self.b - np.dot(self.A, x0)
        d = -r
        for k in range(self.max_iter):
            z = np.dot(self.A, d)
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
