import numpy as np

def bidiagonalize(A):
    """
    Bidiagonalize matrix A using Householder reflections.
    ========================================

    Params: A (m * n)
    
    Returns: 
    """
    m, n = A.shape
    if_transpose = 0
    if m < n:
        B = A.T.copy()
        m, n = B.shape
        if_transpose = 1
    else:
        B = A.copy()
    
    U = np.eye(m)
    V = np.eye(n)

    for k in range(min(m-1, n)):
        # left Householder reflection
        B_sub = B[k:, k:]
        e_k = np.eye(m-k)[:,0]
        u = B_sub[:,0] - np.sign(B_sub[0,0]) * np.linalg.norm(B_sub[:,0]) * e_k
        v = u / np.linalg.norm(u)

        B_sub -= 2 * np.outer(v, v @ B_sub)
        U[:, k:] -= 2 * np.outer(U[:,k:] @ v, v)

        # right Householder reflection, for n-2 times
        if k < n-2:
            B_sub = B[k:, k+1:]


            e_k = np.eye(n - (k + 1))[0,:]

            u = B_sub[0,:] - np.sign(B_sub[0,0]) * np.linalg.norm(B_sub[0,:]) * e_k
            v = u / np.linalg.norm(u)

            B_sub -= 2 * np.outer(B_sub @ v, v)
            V[:,k+1:] -= 2 * np.outer(V[:,k+1:] @ v, v)
        
    if if_transpose:
        return V, B.T, U
    else:
        return U, B, V 


# Test the implementation for all cases
A1 = np.random.randn(4, 6)
U, B, V = bidiagonalize(A1)
print(np.allclose(U.T @ U, np.eye(4), atol=1e-8))
print(np.allclose(V.T @ V, np.eye(6), atol=1e-8))
print(np.allclose(U @ B @ V.T, A1, atol=1e-8))


