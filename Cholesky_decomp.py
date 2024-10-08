import numpy as np


def cholesky_decomp(A):
    n = len(A)
    L = np.zeros(shape=(n, n))
    L[0,0] = np.sqrt(A[0,0])
    for i in range(1, n):
        for j in range(i+1):
            #print(i)
            if i == j:
                L[i, i] = np.sqrt(A[i, i] - np.sum(L[i,:j]**2)) 
            else:
                L[i, j] = 1/L[j, j] * (A[i, j] - np.sum(L[i,:j]*L[j,:j]))
    return L
                
if __name__ == '__main__':
    A1 = np.array([[2, -1, 0, 0],
                  [-1, 2, -1, 0],
                  [0, -1, 2, -1],
                  [0, 0, -1, 2]])
    
    A2 = np.array([[1, 1/2, 1/3, 1/4],
                   [1/2, 1/3, 1/4, 1/5],
                   [1/3, 1/4, 1/5, 1/6],
                   [1/4, 1/5, 1/6, 1/7]])
    
    L1 = cholesky_decomp(A1)
    L2 = cholesky_decomp(A2)
    print("The L matrix of A1 is:\n", L1)
    print("The L matrix of A2 is:\n",L2)
    