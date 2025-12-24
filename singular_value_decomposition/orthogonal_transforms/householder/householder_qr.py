import numpy as np
import math 

def householder_QR_decomposition(A):
    m,n = A.shape
    Q = np.eye(m)
    R = A.copy() 
    
    # m = n, m > n and m < n all works here.
    for k in range(min(m-1, n)):     
        R_sub = R[k:,k:]
        Q_sub = Q[:,k:]
        e_k = np.eye(m-k)[:,0]

        # Compute the Householder vector: H = I - 2vv^T
        u = R_sub[:,0] - np.linalg.norm(R_sub[:,0]) * np.sign(R_sub[0][0]) * e_k 
        v = u / np.linalg.norm(u)

        # no need to construct H explicitly, solve R and Q iteratively.
        R_sub -= 2* np.outer(v, v @ R_sub)
        Q_sub -= 2 * np.outer(Q_sub @ v, v)
        
    return Q, R 

# Test the implementation for all cases
A1 = np.random.randn(1000, 1000)
Q, R = householder_QR_decomposition(A1)

print(np.allclose(Q.T @ Q, np.eye(1000), atol=1e-8)) 
print(np.allclose(Q @ R, A1, atol=1e-8)) 

# A2 = np.random.randn(5, 3)
# Q, R = householder_QR_decomposition(A2)
# print(np.allclose(Q.T @ Q, np.eye(5), atol=1e-8))
# print(np.allclose(Q @ R, A2, atol=1e-8))

# A3 = np.random.randn(3, 5)
# Q, R = householder_QR_decomposition(A3) 
# print(np.allclose(Q.T @ Q, np.eye(3), atol=1e-8))
# print(np.allclose(Q @ R, A3, atol=1e-8))