import numpy as np
import time

def backtrack(x0, fx0, dfx0, points, b):
    eta = 2
    l = 1
    maxit = 100
    
    x = x0
    l_prev = l
    for i in range(0, maxit):        
        l = (eta**i) * l
        x = x0 - (1/l) * dfx0
        
        diff = x - points
        diff_norm_2 = np.sum(diff**2, axis=1)
        g = diff_norm_2**0.5
        mask = g < b
        g[mask] = diff_norm_2[mask] / (2 * b[mask]) + b[mask] / 2
        fs = np.sum(g)

        if fs > fx0 - (1/(2*l)) * np.sum(dfx0**2):
            return l_prev



def accelerated_weiszfeld(points, debug=False):
    # find minimum anchor point
    min_f, min_s, min_p = None, None, None

    fs = []

    for p in points:
        sod = np.sum((points - p)**2, axis=1)**0.5
        f = sum(sod)
        fs.append(f)
        if min_f is None or f < min_f:
            min_f, min_s, min_p = f, sod, p

    fs = np.array(fs)

    # check if minimum is a solution
    mask = np.logical_or(points[:,0] != min_p[0], points[:,1] != min_p[1])

    no_min_points = points[np.array(mask)]
    min_s = min_s[mask]

    s = np.sum((min_p - no_min_points) / min_s[:,None], axis=0)
    norm_s = np.sum(s**2)**0.5

    if norm_s <= 1:
        # if it is a solution, exit with no descent direction
        return min_p, None, None, None

    d = -s / norm_s
    l = np.sum(1 / min_s[:,None])
    t = (norm_s - 1) / l

    w = min_p + d*t
    sod_w = np.sum((points - w)**2, axis=1)**0.5
    fw = sum(sod_w)

    b = fs - fw
    
    start_time = time.time()

    max_error = 1e-5
    ext_condition = True
    
    result = []

    y = w
    x = w

    cnt = 0
    
    while ext_condition:
        diff = y - points
        diff_norm = np.sum(diff**2, axis=1)**0.5

        f = diff_norm.sum()
        result.append(f)

        grads = diff / diff_norm[:,None]
        mask = diff_norm < b
        grads[mask] = diff[mask] / b[mask][:,None]
        grad = np.sum(grads, axis=0)

        if debug:
            print("f={} x={} grad={}".format(f, y, grad))

        ls = backtrack(y, f, grad, points, b)

        x_old = x
        x = y - (1/ls) * grad

        t_old = t
        t = (1 + (1 + 4 * t**2)**0.5)*0.5

        y_old = y
        y = x + ((t_old - 1)/t)*(x - x_old)

        ext_condition = (np.abs(y - y_old) > max_error).any()

    return np.array(result), time.time() - start_time