import time
import numpy as np
from common import nonmonotone_line_search, get_min_anchor_direction_and_stepsize, get_start_point, f_grad_hess, armijo_line_search

def newton(points, debug=True):
    # ap, d, t = get_min_anchor_direction_and_stepsize(points)
    start_time = time.time()

    # ap, is_optimal, f, d, t = get_start_point(points)
    # if d is None:
    #     return np.array([f]), time.time() - start_time, 0
    # x = ap + t*d

    x = np.average(points, axis=0)

    start_iters_time = time.time()

    eps = 1e-5
    p = 1e-4

    gamma = 1

    result = []
    
    max_error = 1e-5
    ext_condition = True
    maxiters = 100
    prev_x = x
    
    while ext_condition and len(result) < maxiters:
        f, grad, hess = f_grad_hess(x, points)

        result.append(f)

        if debug:
            print("f={} x={} grad={}".format(f, x, grad))

        norm_grad = np.sum(grad**2)**0.5

        if norm_grad <= eps:
            break

        # t d = −∇^2f(x_k)^(−1) * ∇f(x_k)
        invh = np.linalg.inv(hess)
        d = - np.matmul(invh, grad)

        # compute stepsize with armijo backtracking
        # condition: f (x_k + γ_k * d_k) ≤ f(x_k) + σ * γ_k * <∇f(x_k), d_k>
        # gamma, x = armijo_line_search(x, d, f, grad, gamma, points)

        prev_x = x
        gamma, x, f = nonmonotone_line_search(points, x, d, result)

        ext_condition = (np.abs(x - prev_x) > max_error).any()

    iter_time = time.time() - start_iters_time

    return np.array(result), time.time() - start_time, iter_time
