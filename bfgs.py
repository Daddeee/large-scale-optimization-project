import time
import numpy as np
from common import get_start_point, f_grad, armijo_line_search

def bfgs(points, debug=False):
    ap, is_optimal, f, d, t = get_start_point(points)
    
    start_time = time.time()

    if d is None:
        raise ValueError("anchor point", ap, "is the solution.")

    x = ap + t*d
    H = np.identity(points.shape[1])

    eps = 1e-5
    gamma = 1

    f, grad = f_grad(x, points)

    result = []

    while True:
        result.append(f)

        if debug:
            print("f={} x={} grad={}".format(f, x, grad))

        norm_grad = np.sum(grad**2)**0.5
        if norm_grad <= eps:
            break

        d = -np.matmul(H, grad)
        
        prev_x = x
        prev_grad = grad

        gamma, x = armijo_line_search(x, np.asarray(d).reshape(-1), f, grad, gamma, points)
        f, grad = f_grad(x, points)

        y = np.matrix(grad - prev_grad).T
        s = np.matrix(x - prev_x).T

        ys = (y.T * s).item()

        if ys == 0:
            break

        Hy = H * y
        yHy = (y.T * Hy).item()
        HYS = Hy * s.T
        H = H + (ys + yHy)*(s*s.T)/(ys**2) - (HYS + HYS.T) / ys

    return np.array(result), time.time() - start_time