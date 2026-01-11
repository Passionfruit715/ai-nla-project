""" Golub-Reinsch SVD implementation.  keyword: bidiagonalization, implicit shifted QR iterations, bulge chasing, deflation(divide & conquer), wilkinson shift.  """

import numpy as np  
import math
from ...orthogonal_transforms.bidiagonalization import bidiagonalize
np.random.seed(0)

EPS = np.finfo(float).eps
UNFL = np.finfo(float).tiny
MAXITER = 100
MEIGTH = -0.125


def btB_2x2_abc(d, e, l, r, pattern="topdown"):
    """
    Return a,b,c for the 2x2 principal submatrix of B^T B.
    B is upper bidiagonal with diag d[0..n-1] and superdiag e[0..n-2].
    Subproblem indices: [l..r].
    
    topdown: use B^TB
    bottomup: use BB^T
    """
    if r-l < 1:
        return None
    if pattern == "topdown":
        i = r - 1
        # (i,i), (i,i+1), (i+1,i+1) of B^T B
        a = d[i]**2 + (e[i-1]**2 if i-1 >= l else 0.0)
        b = d[i] * e[i]
        c = d[i+1]**2 + e[i]**2
        return a, b, c
    if pattern == "bottomup":
        #i = l
        #a = d[i]**2 + e[i]**2
        #b = d[i+1] * e[i]
        #c = d[i+1]**2 + (e[i+1]**2 if i+1<r else 0.0)

        i = r-1
        a = d[i]**2 + e[i]**2
        b = d[i+1]*e[i]
        c = d[i+1]**2
        return a, b, c

    raise ValueError("end must be 'trailing' or 'leading'")


def wilkinson_from_abc(a, b, c):
    """
    STABLE version of wilkinson shift computation, used for accelerate QR iterations. 
    suppose the right bottom 2 x 2 submatrix of B^TB is M = [[a, b],[b, c]]
    shift = the eigenvalue of M that is closest to entry c. 

    Args: 
    d: main diagonal entries.
    e: superdiagonal entries.
    l: the left start index of the submatrix
    r: the right start index of the submatrix

    Return:
    the wilkinson shift. 
    """
    delta = 0.5 * (a - c)
    sgn = 1.0 if delta >= 0 else -1.0
    # use hypot to maintain numerical stability.
    denom = abs(delta) + np.hypot(delta, b)
    
    if denom == 0:
        return c

    shift = c - sgn * b**2 / denom
    return shift

def wilkinson_shift(d,e,l,r, pattern="topdown"):
    abc = btB_2x2_abc(d, e, l, r, pattern=pattern)
    if abc is None:
        return 0.0
    return wilkinson_from_abc(*abc)


def givens_rotation(a, b):
    """
    apply givens matrix to [a, b], such that [[c, s],[-s, c]] @ [a,b]^T = [r, 0]^T
    """
    if b == 0.0:
        return 1.0, 0.0, a
    if a == 0:
        return 0.0, np.sign(b), abs(b)
    
    r = np.hypot(a, b)
    c = a / r
    s = b / r
    return c, s, r

def _assert_near_zero(x, scale, k=50, msg=""):
    """
    assert a variable is eliminated to near zero under appropriate scale.
    """
    tol = k * EPS * scale
    if abs(x) <= tol:
        print(msg)
        return True
    else:
        return False

def apply_left_givens(c_l, s_l, dk, dk1, ek, ek1, bg_old, bg_new, transpose = False):
    """
    applys the givens rotation on the left: G @ [a,b]^T = [r, 0]^T
    """
    # compute the new matrix value: matrix multiplication.    
    if transpose == True:
        s_l = -s_l
    dk_new = c_l*dk + s_l*bg_old
    ek_new = c_l*ek + s_l*dk1
    bulge2 = c_l*bg_new + s_l*ek1

    bg_old = -s_l*dk + c_l*bg_old

    dk1_new = -s_l*ek + c_l*dk1
    ek1_new = -s_l*bg_new + c_l*ek1
 
    return dk_new, dk1_new, ek_new, ek1_new, bulge2, bg_old

def apply_right_givens(c_r, s_r, ek, ek1, dk1, dk2, bg_old, bg_new, transpose = False):
    """
    applys the givens rotation on the right: [a,b] @ G^T = [r, 0]
    """
    # compute the new matrix value: just matrix multiplication.
    if transpose == True:
        s_r = -s_r
    ek_new = c_r*ek + s_r*bg_old
    dk1_new = c_r*dk1 + s_r*ek1
    bulge2 = c_r*bg_new + s_r*dk2  

    bg_old = -s_r*ek + c_r*bg_old  

    ek1_new = -s_r*dk1 + c_r*ek1
    dk2_new = -s_r*bg_new + c_r*dk2

    return ek_new, ek1_new, dk1_new, dk2_new, bulge2, bg_old


def apply_givens_to_cols(U, c, s, j, transpose=False):
    """
    apply givens to left orthogonal matrices U, construct complete SVD. 
    !extract cols are more expensive than rows, see if optimize later.
    """
    if transpose == True:
        s = -s
    uj = U[:,j].copy()
    uj1 = U[:,j+1].copy()

    U[:,j] = c*uj + s*uj1
    U[:,j+1] = -s*uj + c*uj1

def bulge_chasing_top_down(d, e, l, r, mu, U = None, V= None):
    """
    implicit shifted QR iterations. 

    Args:
    mu: wilkinson shift of B^TB.
    U, V: left/right orthogonal matrix of bidiagonalization.
    """
    # first givens rotataion, use the shift of B^TB to start bulge chasing.
    bulge = 0.0
    a = d[l]**2 - mu
    b = d[l] * e[l]
    c_r,s_r,_ = givens_rotation(a, b)
    
    _, e[l], d[l], d[l+1], bulge,__ = apply_right_givens(c_r, s_r, 0.0, e[l], d[l], d[l+1], bulge, 0.0, transpose=False) 
    apply_givens_to_cols(V, c_r, s_r, 0)

    # main iterations for bulge chasing, apply left and right givens rotations alternately.
    for k in range(l, r-1):
        c_l, s_l, _ = givens_rotation(d[k], bulge)
        d[k], d[k+1], e[k], e[k+1], bulge, _ = apply_left_givens(c_l, s_l, d[k], d[k+1], e[k], e[k+1], bulge, 0.0, transpose=False)
        apply_givens_to_cols(U, c_l, s_l, k)

        
        c_r, s_r, _ = givens_rotation(e[k], bulge)
        e[k], e[k+1], d[k+1], d[k+2], bulge, __= apply_right_givens(c_r, s_r, e[k], e[k+1], d[k+1], d[k+2], bulge, 0.0, transpose=False)
        apply_givens_to_cols(V,c_r, s_r, k+1)

    # last givens rotation for the right bottom corner 2*2 submatrix to exit bulge chasing, bulge should be 0 in the end.  
    c_l, s_l, _ = givens_rotation(d[r-1], bulge)
    d[r-1], d[r], e[r-1], _, bulge, __ = apply_left_givens(c_l, s_l, d[r-1], d[r], e[r-1], 0.0, bulge, 0.0, transpose=False)
    apply_givens_to_cols(U,c_l, s_l, r-1)

    return d, e, U, V



def bulge_chasing_bottom_up(d, e, l, r, mu, U= None, V=None):
    """
    chasing the bulge from bottom to top.
    """

    B = np.diag(d) + np.diag(e, 1)
    u0, s0, v0 = np.linalg.svd(B)

    bulge = 0.0
    a = d[r]**2 - mu
    b = d[r] * e[r-1]
    c_l,s_l, _ = givens_rotation(a, b)

    d[r-1], d[r], e[r-1], _, __, bulge = apply_left_givens(c_l, s_l, d[r-1], d[r], e[r-1], 0.0, 0.0, bulge, transpose=True)
    apply_givens_to_cols(U, c_l, s_l, r-1, transpose=True)


    for k in range(r, l+1, -1):
        c_r, s_r, _ = givens_rotation(d[k], bulge)
        e[k-2], e[k-1], d[k-1], d[k], _, bulge = apply_right_givens(c_r, s_r, e[k-2], e[k-1], d[k-1], d[k], 0.0, bulge, transpose=True)
        apply_givens_to_cols(V, c_r, s_r, k-1, transpose=True)

        c_l, s_l,_ = givens_rotation(e[k-1], bulge)
        d[k-2], d[k-1], e[k-2], e[k-1], _, bulge = apply_left_givens(c_l, s_l, d[k-2], d[k-1], e[k-2], e[k-1], 0.0, bulge, transpose=True)
        apply_givens_to_cols(U, c_l, s_l, k-2, transpose=True)

    c_r, s_r, _ = givens_rotation(d[l+1], bulge)
    _, e[l], d[l], d[l+1], __, bulge = apply_right_givens(c_r, s_r, 0.0, e[l], d[l], d[l+1], 0.0, bulge, transpose=True)
    apply_givens_to_cols(V, c_r, s_r, l, transpose=True)

    B = np.diag(d) + np.diag(e, 1)
    u, s, v = np.linalg.svd(B)
    print(f" 4 {np.max(abs(s - s0))}")

    return d, e, U, V

def deflate_threshold(d, e, l, r):
    """
    compute the threshold for entry on superdiagonal(e) should deflate or not, LAPACK style.
    """
    d = np.asarray(d, dtype=float)
    e = np.asarray(e, dtype=float)
    n = d.size

    smax = np.max(np.abs(d))
    if e.size:
        smax = max(smax, np.max(np.abs(e)))
    
    tolmul = max(10.0, min(100.0, EPS**MEIGTH))
    tol = tolmul*EPS
    
    noise_floor = float(MAXITER) * float(n) * float(n) * UNFL
    # estimate the scale of minimum singular value.
    
    sminoa = float(abs(d[l]))
    if sminoa == 0.0:
        thresh = max(0.0, noise_floor)
        return thresh
    
    mu = sminoa
    for i in range(l+1, r):
        mu = abs(d[i]) * (mu / (mu + abs(e[i-1])))
        
        sminoa = min(mu, sminoa)
        if sminoa == 0.0:
            break
    sminoa /= math.sqrt(float(n))
    thresh = max(tol*sminoa, noise_floor)
    return thresh

def sweep_deflate(d,e,l,r,threshold):
    """
    sweep the whole superdiagonal, and set all deflated entries to 0.0.
    """
    for k in range(l, r):
        if e[k] <= threshold:
            e[k] = 0.0

def find_active_block(e, r):
    """
    from bottom to top, find a active submatrix to continue perform bulge chasing.
    """
    l = r 
    while l > 0 and e[l-1] != 0.0:
        l -= 1
    return l


def deflation(d, e, U, V):
    """
    deflation to smaller submatrices, thus no need to perform QR iterations for the whole matrix, divide and conquer.
    """
    r = len(d)-1
    l = 0
    
    i = 0
    while r > 0:
        # the bottom submatrix only left with one entry, deflate. gradually move r up. 
        while r > 0 and e[r-1] == 0.0:  # add r > 0 to aviod e[-1]
            r -= 1

        #breakpoint()
        
        if r == 0:
            break

        l  = find_active_block(e, r)
        threshold = deflate_threshold(d,e,l,r)
        
        if abs(d[l]) <= abs(d[r]):
            mu = wilkinson_shift(d,e,l,r, pattern="bottomup")
            if _assert_near_zero(mu, abs(d[r])**2, msg="bottomup mu too small"):
                mu = 0.0
            d,e,U,V = bulge_chasing_bottom_up(d,e,l,r,mu,U,V)
        else:
            mu = wilkinson_shift(d,e,l,r, pattern="topdown")
            if _assert_near_zero(mu, abs(d[l])**2, msg="topdown mu too small"):
                mu = 0.0
            d, e, U, V = bulge_chasing_top_down(d, e, l, r, mu, U, V)
          
        sweep_deflate(d,e,l,r, threshold)

        #print(f"this is {i}th iteration")
        i+=1
    
    return d, e, U, V


def golub_reinsch_svd(A):
    """
    the main loop for singular value decomposition.
    """

    U, B, V, if_transpose =  bidiagonalize(A)
    
    d = np.diag(B).copy()
    e = np.diag(B, k=1).copy()
    d, e, U, V = deflation(d, e, U, V)
 
    # if transpose, means the result is svd of A.T. 
    # need to reassign U and V coresspondely.  
    # svd process: A.T = U @ d @ V.T
    # A = V @ D @ U.T 
    # thus, we assign u_out = V, vt_out = U.T
    if if_transpose:
        u_out = V
        vt_out = U.T
    else:
        u_out = U
        vt_out = V.T

    # sort the d entries in descent order, means singular value in descent order.
    sign = np.sign(d)
    sign[sign == 0] = 1

    d = abs(d)
    vt_out *= sign[:, None]

    idx = np.argsort(d)[::-1]
    d_out = d[idx]
    u_out = u_out[:,idx]
    vt_out = vt_out[idx,:]

    return d_out, e, u_out, vt_out


if __name__ == "__main__":
    ## test case 1 for SVD, m < n: 
    #A1 = np.random.randn(4,6)
    #d1, _, u1, v1 = golub_reinsch_svd(A1)
    #
    ## verify using standard package svd algorithms.
    #u, s1, vh = np.linalg.svd(A1, compute_uv=True, full_matrices=False)
    #
    #A_hat1 = u1 @ np.diag(d1) @ v1
    #print(np.linalg.norm(A1 - A_hat1) / np.linalg.norm(A1))
    #print(np.allclose(d1, s1, atol=1e-8))

    ## test case 2: m > n

    #A2 = np.random.randn(6,4)
    #d2, _, u2, v2 = golub_reinsch_svd(A2)
    #
    ## verify using standard package svd algorithms.
    #u, s2, vh = np.linalg.svd(A2, compute_uv=True, full_matrices=False)
    #
    #A_hat2 = u2 @ np.diag(d2) @ v2
    #print(np.linalg.norm(A2 - A_hat2) / np.linalg.norm(A2))
    #print(np.allclose(d2, s2, atol=1e-8))


    ## test case 3
    U0, _ = np.linalg.qr(np.random.randn(6,6))
    V0, _ = np.linalg.qr(np.random.randn(6,6))

    
    # test case 4: 
    #s = np.array([3.5, 2.1, 3.22, 4.13, 5.35])
    #e = np.array([0.5, 0.4, 0.3, 0.2])
    s = np.array([5.5, 2.31, 3.25, 1e-8, 1e-10, 1e-12])
    A = U0[:, :6] @ np.diag(s) @ V0.T
    #A = np.random.randn(1000, 1000)
    #d,e,u,v = deflation(s, e, np.eye(5), np.eye(5))

    d, e, U, Vt = golub_reinsch_svd(A)
    #
    #r = len(d)
    #U_r = U[:, :r]
    #Vt_r = Vt[:r,:]
    Ahat = U @ np.diag(d) @ Vt
    rel_recon = np.linalg.norm(A - Ahat) / np.linalg.norm(A)
    print("rel_recon =", rel_recon)

    Sigma = np.diag(d)
    print("diag-res =", np.linalg.norm(U.T @ A @ Vt.T - Sigma) / np.linalg.norm(Sigma))

    u, s, vh = np.linalg.svd(A, compute_uv=True)


