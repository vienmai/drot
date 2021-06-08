import numpy as np
import scipy as sp
import numpy.linalg as nla
from time import time

def drot(init, proxf, proxg, b, max_iters=10, **kwargs):
    # Stopping parameters
    eps_abs = kwargs.pop("eps_abs", 1e-6)   
    eps_rel = kwargs.pop("eps_rel", 1e-8)   

    # Stepsize parameters
    step = kwargs.pop("step", 1.0)  
    adapt_stepsize = kwargs.pop("adapt_stepsize", True)
    incr = kwargs.pop("incr", 2.0)
    decr = kwargs.pop("decr", 2.0)
    mu = kwargs.pop("mu", 10.0)
    max_step = kwargs.pop("max_step", 1.0)
    min_step = kwargs.pop("min_step", 1e-4)

    # Overelaxation parameter
    lamda = kwargs.pop("relaxation", 1)    

    # Restart parameters
    restart = kwargs.pop("restart", True)
    adapt_restart = kwargs.pop("adapt_restart", True)

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)   
    
    assert (max_step >= min_step), "Maximum stepsize must be larger than minimum one."
    assert (lamda > 0 and lamda < 2), "Relaxation parameter must be in (0,2)."

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("----------------------------------------------------")

    k = 0
    done = False
    y = np.array(init, order='F')
    r_primal = np.zeros(max_iters)
    r_dual = np.zeros(max_iters)
    r_full = np.infty
    r_full0 = 0.0
    m, n = y.shape
    e = np.ones(n)
    f = np.ones(m)
    if compute_r_primal:
        Az = np.zeros(b.shape)
        
    start = time()
    while not done:
        x = proxg(y)
        z = 2 * x - y
        proxf(z, step)
        x -= z # overide x by x - z 
        y -= lamda * x
        
        assert x.flags['F_CONTIGUOUS']   
        assert y.flags['F_CONTIGUOUS']   
        assert z.flags['F_CONTIGUOUS']   

        if compute_r_primal:
            Az =np.vstack((z.dot(e), z.T.dot(f)))
            r_primal[k] = nla.norm(Az - b)
        if compute_r_dual:
            r_dual[k] = nla.norm(x, ord='fro') / step
        if compute_r_primal or compute_r_dual:
            r_full = np.sqrt((r_primal[k]**2 + r_dual[k]**2))
            if k == 0: 
                r_full0 = r_full
        
        if adapt_stepsize:
            if (r_primal[k] > mu * r_dual[k]) and (step * incr <= max_step):
                step *= incr   
                print("Stepsize increased, new value is ", step) 
            elif (r_dual[k] > mu * r_primal[k]) and (step / decr >= min_step):
                step /= decr
                print("Stepsize decreased, new value is ", step)

        if (k % print_every == 0 or k == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(k).rjust(6), 
                                        format(res, ".2e").ljust(10),
                                        format(r_primal[k], ".2e").ljust(11), 
                                        format(r_dual[k], ".2e").ljust(9),
                                        format(time() - start, ".2e").ljust(8)))
        k += 1
        done = k >= max_iters or (r_full <= eps_abs + eps_rel * r_full0)
    
    end = time()
    print("Solve time: ", end - start)
    
    return {"sol":          z, 
            "primal":       np.array(r_primal[:k]), 
            "dual":         np.array(r_dual[:k]), 
            "num_iters":    k, 
            "solve_time":   (end - start)}