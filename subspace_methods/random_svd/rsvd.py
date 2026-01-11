"""
Implementation of randomized SVD.
"""
from ...orthogonal_transforms.householder.householder_qr import householder_QR_decomposition
from ...exact_methods.gr_svd import golub_reinsch_svd
import numpy as np


def gaussian_test_matrix(n, l, rng=None):
    """
    generate a n * l gaussian_test_matrix with i.i.d. N(0, 1) entries

    Args: 
    rng: random number generator: 
    rng usage: np.random.default_rng(6643), not conflict with global randomness
    """
    if rng == None:
        return np.random.randn(n, l)
    return rng.standard_normal((n, l))


def r_svd(A, k, p=10, q=1, rng=None):
    """
    ramdomized svd: 

    notes: CSR/CSC matrix representation.
    
    Args:
    k: the low rank we wish to keep.
    p: oversampling dim
    q: power iteration times.
    """
    n = A.shape[1]
    Omega = gaussian_test_matrix(n, (k + p), rng)
    
    Y = A @ Omega  # TODO what if A larger than memory?
    
    # power iterations
    for _ in range(q):
        Q, _ = householder_QR_decomposition(Y, mode="reduced")
        Y = A @ (A.T @ Q)
    
    Q, _ = householder_QR_decomposition(Y, mode="reduced")

    # project Y into Q subspace, B = Q.T A
    B = Q.T @ A

    # perform svd in subspace B
    d, _, U_tilde, Vt = golub_reinsch_svd(B)
    #U_tilde, d, Vt = np.linalg.svd(B)

    # transform U back
    U = Q @ U_tilde

    return U[:, :k], d[:k], Vt[:k, :]

if __name__ == "__main__":
    A1 = np.random.randn(4, 6)
    u,s,vt = r_svd(A1, 2)


    


