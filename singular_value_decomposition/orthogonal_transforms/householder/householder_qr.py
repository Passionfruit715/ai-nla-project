import numpy as np

def householder_QR_decomposition(A, mode="reduced"):
    """
    first loop: accumulate v and update R.
    second loop: update Q backwards, H0 @ H1 @ H2.....Hk @ I, first compute Hk @ I, and move backwards.
    """
    m,n = A.shape 
    R = A.copy()
    vs  = []  # save all the v vectors to get Q later.
    k = n if m > n else m-1 # number of householder trans we will perform, for all cases.
    for i in range(k):     
        x = R[i:, i]
        normx = np.linalg.norm(x)
        if normx == 0.0:    # if x is a zero vectors, skip.
            vs.append(None)
            continue
        
        s = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += s*normx
        v = v / np.linalg.norm(v)
        vs.append(v)    # store V

        R[i:,i:] -= 2* v[:, np.newaxis] * (v @ R[i:,i:]) # update R using broadcast.

    # initialize Q separately.
    if mode == "reduced":
        Q = np.eye(m, min(m,n))
    elif mode == "full":
        Q = np.eye(m)
    else:
        raise ValueError("mode must be 'reduced' or 'full'")
    
    # update Q from backwards.
    for j in range(k-1, -1, -1):
        v = vs[j]
        if v is None:
            continue
        Q[j:,:] -= 2 * v[:, np.newaxis] * (v @ Q[j:,:])  # broadcast


    if mode == "reduced":
        return Q, R[:min(m,n), :]
    return Q, R 


def test_qr(A):
    Q, R = householder_QR_decomposition(A)
    
    # reconstruct error
    err1 = np.linalg.norm(A - Q@R) / np.linalg.norm(A)

    k = Q.shape[1]
    err2 = np.linalg.norm(Q.T@Q - np.eye(k))
    err3 = np.linalg.norm(np.tril(R[:, :k], -1))
    
    # orthogonal transform do not change norm. 
    err4 = abs(np.linalg.norm(A, 'fro') - np.linalg.norm(R, 'fro')) / np.linalg.norm(A, 'fro')

    return err1, err2, err3, err4

if __name__ == "__main__":
    # random generate matrix
    m = 20
    n = 15
    A1 = np.random.randn(m, n)
    
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, -12, min(m, n))
    A2 = U[:, :len(s)] @ np.diag(s) @ V[:len(s), :]
    
    # rank - deficient
    A3 = A1.copy()
    A3[:, -1] = A3[:, 0] + 1e-12 * A3[:, 1]

    m1 = 1000 
    n1 = 5
    A4 = np.random.randn(m1, n1)

    print(test_qr(A1))
    print(test_qr(A2))
    print(test_qr(A3))
    print(test_qr(A4))




