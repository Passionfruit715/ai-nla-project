""" Golub-Reinsch SVD implementation.  keyword: bidiagonalization, implicit shifted QR iterations, bulge chasing, deflation(divide & conquer), wilkinson shift.  """

import numpy as np  
from phase1_nla.bidiagonalization import bidiagonalize
np.random.seed(0)

EPS = np.finfo(float).eps
debug = True


def wilkinson_shift(d, e, l, r):
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
    # if deflated into 1*1 submatrix, no shift needed.
    if r - l < 1:
        return 0.0
    
    # if 2*2 submatrix occurs during delation, add condition r-2 >= l to correct the computation.
    a = d[r-1]**2 + (e[r-2]**2 if r-2 >= l else 0.0)
    b = d[r-1] * e[r-1]
    c = d[r]**2 + e[r-1]**2

    delta = 0.5 * (a - c)
    sgn = 1.0 if delta >= 0 else -1.0
    # use hypot to maintain numerical stability.
    denom = abs(delta) + np.hypot(delta, b)
    
    if denom == 0:
        return c

    shift = c - sgn * b**2 / denom
    return shift

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

def _assert_bulge_near_zero(x, scale, k=50, msg=""):
    """
    assert bulge is eliminated to near zero.
    """
    tol = k * EPS * max(1.0, scale)
    assert abs(x) < tol, f"{msg} |x| = {abs(x)}, tol = {tol}, x = {x}"

def apply_left_givens(dk, dk1, ek, ek1, bulge):
    """
    applys the givens rotation on the left: G @ [a,b]^T = [r, 0]^T
    """
    c_l, s_l, _ = givens_rotation(dk, bulge)

    # compute the new matrix value: matrix multiplication.
    dk_new = c_l*dk + s_l*bulge 
    ek_new = c_l*ek + s_l*dk1
    bulge2 = s_l*ek1

    bg_new = -s_l*dk + c_l*bulge

    if debug:
        _assert_bulge_near_zero(bg_new, abs(dk) + abs(bulge), msg="left givens bg_new not near ~0")

    dk1_new = -s_l*ek + c_l*dk1
    ek1_new = c_l*ek1
    
    return c_l, s_l, dk_new, dk1_new, ek_new, ek1_new, bulge2

def apply_right_givens(ek, ek1, dk1, dk2, bulge):
    """
    applys the givens rotation on the right: [a,b] @ G^T = [r, 0]
    """
    c_r, s_r, _ = givens_rotation(ek, bulge)

    # compute the new matrix value: just matrix multiplication.
    ek_new = c_r*ek + s_r*bulge
    dk1_new = c_r*dk1 + s_r*ek1
    bulge2 = s_r*dk2 

    bg_new = -s_r*ek + c_r*bulge  # bg_new should become 0 or very close to 0.
    if debug:
        _assert_bulge_near_zero(bg_new, abs(ek) + abs(bulge), msg="right givens bg_new not near ~0")

    ek1_new = -s_r*dk1 + c_r*ek1
    dk2_new = c_r*dk2

    return c_r, s_r, ek_new, ek1_new, dk1_new, dk2_new, bulge2

def apply_givens_to_cols():
    """
    apply givens to left and right orthogonal matrices, construct complete SVD.
    """
    return 0

def bulge_chasing(d, e, l, r, mu):
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
    dk, dk1, ek = d[l], d[l+1], e[l]
    
    dk_new = c_r*dk + s_r*ek
    ek_new = -s_r*dk + c_r*ek
    bulge = s_r*dk1
    dk1_new = c_r*dk1

    d[l],d[l+1],e[l] = dk_new, dk1_new, ek_new

    # main iterations for bulge chasing, apply left and right givens rotations alternately.
    for k in range(l, r-1):
        # c and s used later to calculate U and V
        c_l, s_l, d[k], d[k+1], e[k], e[k+1], bulge = apply_left_givens(d[k], d[k+1], e[k], e[k+1], bulge)
        c_r, s_r, e[k], e[k+1], d[k+1], d[k+2], bulge = apply_right_givens(e[k], e[k+1], d[k+1], d[k+2], bulge)

    # last givens rotation for the right bottom corner 2*2 submatrix to exit bulge chasing.  
    c_l, s_l, _ = givens_rotation(d[r-1], bulge)
    dk, dk1, ek, bg = d[r-1], d[r], e[r-1], bulge

    dk_new = c_l*dk + s_l*bg
    ek_new = c_l*ek + s_l*dk1

    bg_new = -s_l*dk + c_l*bg

    if debug:
        _assert_bulge_near_zero(bg_new, abs(d[r-1]) + abs(bulge), msg="last left givens bg_new not near ~0")
    dk1_new = -s_l*ek + c_l*dk1

    d[r-1], d[r], e[r-1] = dk_new, dk1_new, ek_new

    return d, e


def should_deflate(dk, dk1, ek, tau=50):
    """
    entry on superdiagonal(e) should deflate or not.
    """
    tol = tau * EPS * max(1.0, abs(dk) + abs(dk1))
    return abs(ek) <= tol

def sweep_deflate(d,e,l,r,tau=50):
    """
    sweep the whole superdiagonal, and set all deflated entries to 0.0.
    """
    for k in range(l, r):
        if should_deflate(d[k], d[k+1], e[k], tau):
            e[k] = 0.0

def find_active_block(e, r):
    """
    from bottom to top, find a active submatrix to continue perform bulge chasing.
    """
    l = r 
    while l > 0 and e[l-1] != 0.0:
        l -= 1
    return l


def gr_svd(d, e, U=None, V=None):
    """
    deflation, main iteration of Golub Reinsch Singular Value Decomposition.
    """
    r = len(d)-1
    l = 0

    while r > 0:
        # the bottom submatrix only left with one entry, deflate. gradually move r up. 
        while r > 0 and e[r-1] == 0.0:  # add r > 0 to aviod e[-1]
            r -= 1
        
        if r == 0:
            break

        l  = find_active_block(e, r)

        mu = wilkinson_shift(d, e, l, r)
        d, e = bulge_chasing(d, e, l, r, mu)

        sweep_deflate(d,e,l,r)
    
    return d, e




if __name__ == "__main__":
    # test case 1 for bulge chasing. 
    
    # Args
    
    # squre matrix
    A1 = np.random.randn(4,6)
    U, B, V = bidiagonalize(A1)

    
    #k = min(B.shape)
    #Bcore = B[:k,:k]

    d = np.diag(B).copy()
    e = np.diag(B, k=1).copy() # superdiagonal
    
    d_new, e_new = gr_svd(d,e)

    d_new = np.sort(np.abs(d_new))[::-1]

    print(f"d_new is {d_new}")
    print(f"e_new is {e_new}")
    

    # verify using standard package svd algorithms.
    s_numpy = np.linalg.svd(B, compute_uv=False)
    
    print(s_numpy)
    print(np.allclose(d_new, s_numpy, atol=1e-8))




    #norm2_before = sum(di*di for di in d) + sum(ei*ei for ei in e)
    #norm2_after  = sum(di*di for di in d_new) + sum(ei*ei for ei in e_new)
    #assert abs(norm2_after - norm2_before) <= 100*EPS*max(1.0, norm2_before)







