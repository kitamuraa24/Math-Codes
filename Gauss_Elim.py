import numpy as np
import os, sys

def GE(n):
    # Initialize arrays:
    A = np.zeros(shape=(n, n))
    b = np.zeros(n)
    x = np.zeros(n)

    # Populate A and b:
    for i in range(n):
        for j in range(n):
            A[i, j] = (1 + i + 1)**(j-1 + 1)
        b[i] = 1/(i + 1)*((1 + i + 1)**n -1)
    
    # Do GE:
    # Create augmented matrix (M):
    M = np.hstack([A, b.reshape(-1, 1)])

    # Forward elim:
    for i in range(n):
        # Find pivot element
        pivot_idx = np.argmax(np.abs(M[i:,i])) + i

        if pivot_idx != i:
            M[[i, pivot_idx]] = M[[pivot_idx, i]]

        # Elimination:
        for j in range(i+1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
        
    #Back Sub:
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:])) / M[i,i]
    
    return x


if __name__ == '__main__':
    n = np.arange(4, 21)
    text = []
    for n_i in n:
        temp = f"n = {n_i}:\n"
        text.append(temp)
        sol_i = GE(n_i)
        for x in sol_i:
            temp = f"\t{x}\n"
            text.append(temp)
    f = open('Outputs.txt', 'w')
    f.write(''.join(text))
    f.close()