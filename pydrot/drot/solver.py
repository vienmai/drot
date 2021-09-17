import numpy as np
import numpy.linalg as nla
from time import time
from .proximal import trace_nonnegative_prox_nb
from .proximal import apply_adjoint_operator_and_override

def drot(init, C, p, q, **kwargs):
    # Stopping parameters
    max_iters = kwargs.pop("max_iters", 100)
    eps_abs = kwargs.pop("eps_abs", 1e-6)
    eps_rel = kwargs.pop("eps_rel", 1e-15)

    # Stepsize parameters
    step = kwargs.pop("step", 1.0)
    adapt_stepsize = kwargs.pop("adapt_stepsize", True)
    incr = kwargs.pop("incr", 2.0)
    decr = kwargs.pop("decr", 2.0)
    mu = kwargs.pop("mu", 20)
    max_step = kwargs.pop("max_step", 1.0)
    min_step = kwargs.pop("min_step", 1e-4)

    # Restart parameters
    fixed_restart = kwargs.pop("fixed_restart", False)
    milestones = kwargs.pop("milestones", [])
    adapt_restart = kwargs.pop("adapt_restart", False)

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)

    assert (max_step >= min_step), "Invalid range"
    assert C.flags['F_CONTIGUOUS']

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("----------------------------------------------------")

    k = 0
    x = np.array(init, order = 'F')
    m, n = x.shape
    e = np.ones(n)
    f = np.ones(m)
    a1 = x.dot(e)
    b1 = x.T.dot(f)
    b = np.hstack((p, q))
    r_primal = np.zeros(max_iters)
    r_dual = np.zeros(max_iters)
    r_full = np.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        # Implicit F-order for Numba
        trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        a2 = x.dot(e)
        b2 = x.T.dot(f)
        t1 = 2 * a2 - a1 - p
        t2 = 2 * b2 - b1 - q
        c1 = f.dot(t1) / (m + n)
        c2 = e.dot(t2) / (m + n)

        # Broadcasting
        yy = t1 - c1
        xx = t2 - c2
        a1 = a2 - yy - c2
        b1 = b2 - xx - c1
        if compute_r_dual:
            r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
        
        apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)

        if compute_r_primal:
            Ax = np.hstack((a2, b2))
            r_primal[k] = nla.norm(Ax - b)
        if compute_r_primal or compute_r_dual:
            r_full = np.sqrt((r_primal[k]**2 + r_dual[k]**2))
            if k == 0:
                r_full0 = r_full

        if k%10==0 and adapt_stepsize:
            if (r_primal[k] > mu * r_dual[k])  and (step / decr >= min_step):                
                step /= decr
                print("Iteration", k,": Stepsize decreased to ", step) 
            elif (r_dual[k] > mu * r_primal[k])  and (step * incr <= max_step):
                step *= incr 
                print("Iteration", k,": Stepsize increased to ", step)                 

        if (k % print_every == 0 or k == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(k).rjust(6),
                                        format(r_full, ".5e").ljust(10),
                                        format(r_primal[k], ".5e").ljust(11),
                                        format(r_dual[k], ".5e").ljust(9),
                                        format(time() - start, ".2e").ljust(8)))
        k += 1
        done = (k >= max_iters) or (r_full <= eps_abs + eps_rel * r_full0)

    end = time()
    print("Drot terminated at iteration ", k-1)

    trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "primal":       np.array(r_primal[:k]),
            "dual":         np.array(r_dual[:k]),
            "num_iters":    k,
            "solve_time":   (end - start)}

def PDHG(init, proxg, proxh, max_iters=10, **kwargs):
    """
        min_x max_y <Ax, y> + g(x) - h(y) | x \in R^n, y \in R^m
    """
    # Stopping parameters
    eps_abs = kwargs.pop("eps_abs", 1e-6)
    eps_rel = kwargs.pop("eps_rel", 1e-15)

    # Stepsize parameters
    eta = kwargs.pop("eta", 1.0)
    tau = kwargs.pop("tau", 1.0)

    # Overelaxation parameter
    lamda = kwargs.pop("relaxation", 1)

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)

    assert (lamda > 0 and lamda < 2), "Relaxation parameter must be in (0,2)."

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("----------------------------------------------------")

    k = 0
    done = False
    x = np.array(init, order='F')
    y = np.zeros(x.shape[0] + x.shape[1])

    r_primal = np.zeros(max_iters)
    r_dual = np.zeros(max_iters)

    start = time()
    while not done:
        x_new = proxg(x, y, eta)
        y = proxh(x, x_new, y, tau)

        assert x.flags['F_CONTIGUOUS']
        assert x_new.flags['F_CONTIGUOUS']

        # r_primal[k] = nla.norm(x, ord=2)
        r_dual[k] = nla.norm(x_new - x, ord='fro')
        res = np.sqrt((r_primal[k]**2 + r_dual[k]**2))

        x = x_new
        if k == 0:
            res0 = res

        if (k % print_every == 0 or k == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(k).rjust(6),
                                              format(res, ".2e").ljust(10),
                                              format(r_primal[k], ".2e").ljust(11),
                                              format(r_dual[k], ".2e").ljust(9),
                                              format(time() - start, ".2e").ljust(8)))
        k += 1
        # done = (k >= max_iters) or (res <= eps_abs + eps_rel * res0)
        done = k >= max_iters

    end = time()
    print("Solve time: ", end - start)

    return {"sol":          x,
            "primal":       np.array(r_primal[:k]),
            "dual":         np.array(r_dual[:k]),
            "num_iters":    k,
            "solve_time":   (end - start)}
