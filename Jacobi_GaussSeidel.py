import numpy as np

class Solver:
    # TO DO: Replace error with residual, and add convergence tol for actual use
    def __init__(self, A, b, max_iter, xtol, verbose=False, w=None, order=np.inf):
        self.A = A
        self.b = b
        self.max_iter = max_iter + 1
        self.xtol = xtol
        self.n = len(self.b)
        self.verbose = verbose
        self.x = np.array([1, 1, 1])
        self.w = w
        self.order = order

    def Jacobi(self):
        print("\nJacobi Method is used.")
        #Initialize x_k, and x_k+1
        x_k = np.zeros(self.n)
        x_kp1 = np.copy(x_k)
        for k in range(1, self.max_iter):
            for i in range(self.n):
                sum = 0
                for j in range(self.n):
                    if j != i:
                        sum += self.A[i, j]*x_k[j]
                x_kp1[i] = 1/self.A[i,i] * (self.b[i] - sum)
                r_k = np.linalg.norm(self.b - np.dot(self.A, x_kp1), ord=self.order)
            if k == 1:
                r_k_0 = np.copy(r_k)
                r_k_old = np.copy(r_k_0)
            r_k_rel = r_k / r_k_old
            r_k_old = np.copy(r_k_0)
            conv = r_k/r_k_0
            if self.verbose == True:
                print(f"Iteration: {k}, x: {x_kp1}, r_k: {r_k}, rel_r: {r_k_rel}")
            # Set x_k+1 to x_k for next iteration 
            x_k = np.copy(x_kp1)
            if conv < self.xtol:
                print(f"Total Iterations: {k} \nr_k_rel: {r_k_rel}")
                break 
            elif k == self.max_iter - 1:
                print("Max Iterations Hit!")  
        return x_k
    
    def Gauss_Seidel(self):
        print("\nGauss-Seidel Method is used.")
        x_k = np.zeros(self.n)
        for k in range(1, self.max_iter):
            x_kp1 = np.zeros(self.n)
            for i in range(self.n):
                sum1 = np.dot(self.A[i, :i], x_kp1[:i])
                sum2 = np.dot(self.A[i, i+1:], x_k[i+1:])
                x_kp1[i] = 1/self.A[i, i] * (self.b[i] - sum1 - sum2)
            r_k = np.linalg.norm(self.b - np.dot(self.A, x_kp1), ord=self.order)
            if k == 1:
                r_k_0 = np.copy(r_k)
                r_k_old = np.copy(r_k_0)
            r_k_rel = r_k / r_k_old
            r_k_old = np.copy(r_k_0)
            conv = r_k/r_k_0
            if self.verbose == True:
                print(f"Iteration: {k}, x: {x_kp1}, r_k: {r_k}, rel_r: {r_k_rel}")
            x_k = np.copy(x_kp1)
            if conv < self.xtol:
                print(f"Total Iterations: {k}\nr_k_rel: {r_k_rel}")
                break
            elif k == self.max_iter - 1:
                print("Max Iterations Hit!")
        return x_k
    
    def SoR(self):
        print("\nSuccesive Over-Relaxation Method is used.")
        x_k = np.zeros(self.n)
        for k in range(1, self.max_iter):
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    if j != i:
                        s += self.A[i, j] * x_k[j]
                x_k[i] = (1 - self.w) * x_k[i] + (self.w/self.A[i,i]) * (self.b[i] - s)
            r_k = np.linalg.norm(self.A @ x_k - self.b, ord = self.order)
            if k == 1:
                r_k_0 = np.copy(r_k)
                r_k_old = np.copy(r_k_0)
            r_k_rel = r_k / r_k_old
            r_k_old = np.copy(r_k_0)
            conv = r_k/r_k_0
            # r_k_old = np.copy(r_k)
            if self.verbose == True:
                print(f"Iteration: {k}, x: {x_k}, r_k: {r_k}, rel_r: {r_k_rel}")
            if conv < self.xtol:
                print(f"Total Iterations: {k}\nrel_r: {r_k_rel}")
                break
            elif k == self.max_iter -1:
                print("Max Iterations Hit!")
        return x_k       


# if __name__ == "__main__":
#     A = np.array([[3, 1, 0],
#                   [1, 3, 1],
#                   [0, 1, 3]])            
#     b = np.array([4, 5, 4])
#     max_iter = 10
#     w = 9 - 3*np.sqrt(7)
#     solve = Solver(A, b, max_iter, verbose=True, w=w, xtol=1e-6)
#     solve.Jacobi()
#     solve.Gauss_Seidel()
#     solve.SoR()

#Problem 3
    # A = np.array([[4, -1, 0, -1, 0, 0, 0, 0, 0],
    #               [-1, 4, -1, 0, -1, 0, 0, 0, 0],
    #               [0, -1, 4, 0, 0, -1, 0, 0, 0],
    #               [-1, 0, 0, 4, -1, 0, -1, 0, 0],
    #               [0, -1, 0, -1, 4, -1, 0, -1, 0],
    #               [0, 0, -1, 0, -1, 4, 0, 0, -1],
    #               [0, 0, 0, -1, 0, 0, 4, -1, 0],
    #               [0, 0, 0, 0, -1, 0, -1, 4, -1],
    #               [0, 0, 0, 0, 0, -1, 0, -1, 4]])
    # b = np.tile(np.array([0, 0, 1]), 3)

    # max_iter = 100
    # xtol = 1e-2
    # solve = Solver(A, b, max_iter, xtol, verbose=True)
    # solve.Jacobi()
    # solve.Gauss_Seidel()