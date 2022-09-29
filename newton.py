import time
import numpy as np
from common import get_min_anchor_direction_and_stepsize, get_start_point, f_grad_hess, armijo_line_search

def newton(points, debug=False):
    ap, d, t = get_min_anchor_direction_and_stepsize(points)
    # ap, is_optimal, f, d, t = get_start_point(points)

    start_time = time.time()

    if d is None:
        raise ValueError("anchor point", ap, "is the solution.")

    x = ap + t*d

    eps = 1e-5
    p = 1e-4

    gamma = 1

    result = []
    
    while True:
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
        gamma, x = armijo_line_search(x, d, f, grad, gamma, points)

        # gamma, x, f, grad = gnonmonotone_line_search(x, d, result)

    return np.array(result), time.time() - start_time
