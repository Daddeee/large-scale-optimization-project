import time
import numpy as np
from common import nonmonotone_line_search, get_start_point, f_grad, armijo_line_search

def bfgs(points, debug=False):
    start_time = time.time()
    
    # ap, is_optimal, f, d, t = get_start_point(points)
    # if d is None:
    #     return np.array([f]), time.time() - start_time, 0
    # x = ap + t*d

    x = np.average(points, axis=0)

    start_iters_time = time.time()

    H = np.identity(points.shape[1])

    eps = 1e-5
    gamma = 1

    f, grad = f_grad(x, points)

    result = []

    max_error = 1e-5
    ext_condition = True
    maxiters = 100
    prev_x = x

    while ext_condition and len(result) < maxiters:
        result.append(f)

        if debug:
            print("f={} x={} grad={}".format(f, x, grad))

        norm_grad = np.sum(grad**2)**0.5

        if norm_grad <= eps:
            break

        norm_grad = np.sum(grad**2)**0.5
        if norm_grad <= eps:
            break

        d = -np.matmul(H, grad)
        d = np.array(d).flatten()
        
        prev_x = x
        prev_grad = grad

        prev_x = x
        gamma, x, _ = nonmonotone_line_search(points, x, d, result)
        # gamma, x = armijo_line_search(x, np.asarray(d).reshape(-1), f, grad, gamma, points)
        f, grad = f_grad(x, points)

        ext_condition = (np.abs(x - prev_x) > max_error).any()

        y = np.matrix(grad - prev_grad).T
        s = np.matrix(x - prev_x).T

        ys = (y.T * s).item()

        if ys == 0:
            break

        Hy = H * y
        yHy = (y.T * Hy).item()
        HYS = Hy * s.T
        H = H + (ys + yHy)*(s*s.T)/(ys**2) - (HYS + HYS.T) / ys

    iter_time = time.time() - start_iters_time

    return np.array(result), time.time() - start_time, iter_time