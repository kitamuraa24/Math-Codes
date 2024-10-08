import numpy as np

class Solver:
    # TO DO: Replace error with residual, and add convergence tol for actual use
    def __init__(self, A, b, max_iter, xtol, verbose=False, w=None):
        self.A = A
        self.b = b
        self.max_iter = max_iter + 1
        self.xtol = xtol
        self.n = len(self.b)
        self.verbose = verbose
        self.x = np.array([1, 1, 1])
        self.w = w

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
                        sum += A[i, j]*x_k[j]
                x_kp1[i] = 1/A[i,i] * (b[i] - sum)
            # If verbose on
            if self.verbose == True:
                r_k_max = np.max(np.abs(b - np.dot(A, x_kp1)))
                if k == 1:
                    r_k_old = 1
                r_k_maxrel = r_k_max / r_k_old
                r_k_old = np.copy(r_k_max)
                print(f"Iteration: {k}, x: {x_kp1}, r_max: {r_k_max}, rel_r: {r_k_maxrel}")
            # Set x_k+1 to x_k for next iteration 
            x_k = np.copy(x_kp1)
            if r_k_max < self.xtol:
                break   
        return x_k
    
    def Gauss_Seidel(self):
        print("\nGauss-Seidel Method is used.")
        x_k = np.zeros(self.n)
        for k in range(1, self.max_iter):
            x_kp1 = np.zeros(self.n)
            for i in range(self.n):
                sum1 = np.dot(A[i, :i], x_kp1[:i])
                sum2 = np.dot(A[i, i+1:], x_k[i+1:])
                x_kp1[i] = 1/A[i, i] * (b[i] - sum1 - sum2)
            if self.verbose == True:
                r_k_max = np.max(np.abs(b - np.dot(A, x_kp1)))
                if k == 1:
                    r_k_old = 1
                r_k_maxrel = r_k_max / r_k_old
                r_k_old = np.copy(r_k_max)
                print(f"Iteration: {k}, x: {x_kp1}, r_max: {r_k_max}, rel_r: {r_k_maxrel}")
            x_k = np.copy(x_kp1)
            if r_k_max < self.xtol:
                break
        return x_k
    
    def SoR(self):
        print("\nSuccesive Over-Relaxation Method is used.")
        x_k = np.zeros(self.n)
        for k in range(1, self.max_iter):
            for i in range(self.n):
                s = 0
                for j in range(self.n):
                    if j != i:
                        s += A[i, j] * x_k[j]
                x_k[i] = (1 - self.w) * x_k[i] + (self.w/A[i,i]) * (b[i] - s)
            if self.verbose == True:
             ek_max = np.max(np.abs(self.x - x_k))
            if k == 1:
                ek_max_old = 1
            ek_max_rel = ek_max / ek_max_old
            ek_max_old = np.copy(ek_max)
            print(f"Iteration: {k}, x: {x_k}, e_max: {ek_max}, rel_e: {ek_max_rel}")
        return x_k       


if __name__ == "__main__":
    A = np.array([[3, 1, 0],
                  [1, 3, 1],
                  [0, 1, 3]])            
    b = np.array([4, 5, 4])
    max_iter = 10
    w = 9 - 3*np.sqrt(7)
    solve = Solver(A, b, max_iter, verbose=True, w=w, xtol=1e-6)
    solve.Jacobi()
    solve.Gauss_Seidel()
    solve.SoR()

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