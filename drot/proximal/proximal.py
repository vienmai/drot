import numpy as np
import scipy as sp


def primal_dual_linear_prox(x, xnew, y, s, r, step=1.0):
    """
    Compute:
        y_new = argmin_z <b, z> + 1/(2 tau) * ||z - y - tau * A(2* xnew - x)||^2
              = y + tau * A(2* xnew - x) - tau * b
    """
    assert x.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy
    assert xnew.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy

    m, n = x.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == s.size) and (n == r.size), "Dimensions mismatch"

    e = np.ones(n)
    f = np.ones(m)
    xe, xf = apply_operator(e, f, x)
    xnew_e, xnew_f = apply_operator(e, f, xnew)
    y1 = step * (2 * xnew_e - xe - s)
    y2 = step * (2 * xnew_f - xf - r)
    return y + np.hstack((y1, y2))

def primal_dual_trace_nonnegative_prox(x, C, y, step=1):
    assert x.flags['F_CONTIGUOUS']
    assert C.flags['F_CONTIGUOUS']
    m, n = x.shape
    T = x - step * C
    e = np.ones(n)
    f = np.ones(m)
    y1 = y[:m]
    y2 = y[m:]
    apply_adjoint_operator_and_override(e, f, y1, y2, T, -step)
    return np.maximum(T, 0.0, order='F')

def trace_nonnegative_prox(x, C, step=1.0):
    """
        x = argmin_y {trace(C^T x) + (0.5 /step) ||y - x||_{F}^2 | y > = 0}
    """
    assert C.flags['F_CONTIGUOUS']
    assert x.flags['F_CONTIGUOUS']
    if np.isscalar(x):
        x = np.array([[x]])
    return np.maximum(x - step * C, 0.0, out=x, order='F')

def generalized_doubly_stochastic_matrices_projection(A, s, r, step=None):
    assert A.flags['F_CONTIGUOUS']  # ensure we don't make an expensive copy
    T = np.array(A, order='F')
    assert T.flags['F_CONTIGUOUS'] # ensure correctness of blas.dger
    m, n = T.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == s.size) and (n == r.size), "Dimensions mismatch"
    e = np.ones(n)
    f = np.ones(m)
    Te_s = T.dot(e) - s
    Tf_r = T.T.dot(f) - r
    t1 = np.dot(f, Te_s) / (m + n)
    t2 = np.dot(e, Tf_r) / (m + n)
    v1 = Te_s - t1 * f
    v2 = Tf_r - t2 * e
    apply_adjoint_operator_and_override(e, f, v1, v2, T, -1.0/n, -1.0/m)
    return T

def generalized_doubly_stochastic_matrices_projection_(A, s, r):
    T = A.copy()
    m, n = T.shape
    assert (m > 0 and n > 0), "Invalid dimensions"
    assert (m == s.size) and (n == r.size), "Dimensions mismatch"
    e = np.ones(n)
    f = np.ones(m)
    Te_s = T.dot(e) - s
    Tf_r = T.T.dot(f) - r
    t1 = np.dot(f, Te_s) / (m + n)
    t2 = np.dot(e, Tf_r) / (m + n)

    T += (-1.0 / n) * np.outer(Te_s - t1 * f, e)
    T += (-1.0 / m) * np.outer(f, Tf_r - t2 * e)
    return T

def apply_operator(e, f, T):
    return T.dot(e), T.T.dot(f)

def apply_adjoint_operator(e, f, y, x):
    return y.dot(e.T) + f.dot(x.T)

def apply_adjoint_operator_and_override(e, f, y, x, T, alpha=1.0, beta=None):
    """
        T <- T + alpha * y e^T + beta* f x^T
    """
    assert T.flags['F_CONTIGUOUS']
    assert e.size == x.size, "Dimension mismatch"
    assert f.size == y.size, "Dimension mismatch"
    if beta is None:
        beta = alpha
    sp.linalg.blas.dger(alpha, y, e, a=T, overwrite_a=1)
    sp.linalg.blas.dger(beta, f, x, a=T, overwrite_a=1)
    return T

def nonneg_projection(x):
    return np.maximum(x, 0.)

def box_projection(x, lower=-1.0, upper=1.0):
    np.clip(x, lower, upper, out=x)
    return x
