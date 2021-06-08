import numpy as np
import scipy as sp


def primal_dual_linear_prox(x, xnew, r, s, step=1.0):
    assert x.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy
    assert xnew.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy

    m, n = x.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == r.shape[0]) and (n == s.shape[0]), "Dimensions mismatch"

    e = np.ones(n)
    f = np.ones(m)
    xe = x.dot(e)
    xf = x.T.dot(f)
    xnew_e = xnew.dot(e)
    xnew_f = xnew.T.dot(f)
    y1 = step * (2 * xnew_e - xe - r)
    y2 = step * (2 * xnew_f - xf - s)  
    return np.hstack((y1, y2))

def primal_dual_trace_nonnegative_prox(x, C, y, step=1):
    assert x.flags['F_CONTIGUOUS'] 
    assert C.flags['F_CONTIGUOUS']  
    m, n = x.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    
    T = x - step * C
    assert T.flags['F_CONTIGUOUS']  # ensure correctness of blas.dger
    
    e = np.ones(n)
    f = np.ones(m)
    y1 = y[:m]
    y2 = y[m:]
    sp.linalg.blas.dger(-step, y1, e, a=T, overwrite_a=1)
    sp.linalg.blas.dger(-step, f, y2, a=T, overwrite_a=1)
    return T


def trace_nonnegative_prox(x, C, step=1.0):
    """ 
        x = argmin_y {trace(C^T x) + (0.5 /step) ||y - x||_{F}^2 | y > = 0} 
    """
    assert C.flags['F_CONTIGUOUS']
    assert x.flags['F_CONTIGUOUS']
    if np.isscalar(x):
        x = np.array([[x]])
    x -= step * C
    return np.maximum(x, 0.0, out=x, order='F')

def generalized_doubly_stochastic_matrices_projection(A, r, s):
    assert A.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy
    T = np.array(A, order='F') 
    assert T.flags['F_CONTIGUOUS'] # ensure correctness of blas.dger
    m, n = T.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == r.shape[0]) and (n == s.shape[0]), "Dimensions mismatch"
    e = np.ones(n)
    f = np.ones(m)
    Te_s = T.dot(e) - s
    Tf_r = T.T.dot(f) - r
    t1 = np.dot(f, Te_s) / (m + n)
    t2 = np.dot(e, Tf_r) / (m + n)
    v1 = Te_s - t1 * f
    v2 = Tf_r - t2 * e
    sp.linalg.blas.dger(-1.0/n, v1, e, a=T, overwrite_a=1)
    sp.linalg.blas.dger(-1.0/m, f, v2, a=T, overwrite_a=1)
    return T    


def generalized_doubly_stochastic_matrices_projection_(A, r, s):
    T = A.copy()
    m, n = T.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == r.shape[0]) and (n == s.shape[0]), "Dimensions mismatch"
    e = np.ones(n)
    f = np.ones(m)
    Te_s = T.dot(e) - s
    Tf_r = T.T.dot(f) - r
    t1 = np.dot(f, Te_s) / (m + n)
    t2 = np.dot(e, Tf_r) / (m + n)

    T += (-1.0 / n) * np.outer(Te_s - t1 * f, e)
    T += (-1.0 / m) * np.outer(f, Tf_r - t2 * e)
    return T

def nonneg_projection(x):
    return np.maximum(x, 0.)


def box_projection(x, lower=-1.0, upper=1.0):
    np.clip(x, lower, upper, out=x)
    return x
