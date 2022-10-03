import time
import numpy as np
from common import nonmonotone_line_search, get_start_point, f_grad, armijo_line_search

# @profile
def lbfgs(points, debug=False):
    start_time = time.time()
    
    # ap, is_optimal, f, d, t = get_start_point(points)
    # if d is None:
    #     return np.array([f]), time.time() - start_time, 0
    # x = ap + t*d

    x = np.average(points, axis=0)
    m = 20

    Y = np.zeros((len(x), m))
    S = np.zeros((len(x), m))
    rho = np.zeros(m)

    start_iters_time = time.time()

    gamma = 1


    f, grad = f_grad(x, points)
    result = []

    eps = 1e-5
    max_error = 1e-5
    ext_condition = True
    maxiters = 100
    while ext_condition and len(result) < maxiters:
        result.append(f)
        it = len(result)

        if debug:
            print("f={} x={} grad={}".format(f, x, grad))

        norm_grad = np.sum(grad**2)**0.5
        if norm_grad <= eps:
            break

        x_old = x
        grad_old = grad

        q = grad
        if it <= m:
            bound = it - 1
        else:
            bound = m

        alpha = np.zeros(bound)

        for i in reversed(range(bound)):
            alpha[i] = np.dot(S[:, i].T, q)
            q = q - rho[i] * alpha[i] * Y[:, i]

        if it > 1:
            last = min(it - 1, m) - 1
            eta = np.dot(S[:, last], Y[:, last]) / (np.dot(Y[:, last], Y[:, last]))
        else:
            eta = 1

        q_bar = eta*q
        for i in range(bound):
            beta = rho[i] * (alpha[i] - np.dot(Y[:, i].T,q_bar))
            q_bar = q_bar + S[:, i] * beta

        d = -q_bar

        gamma, x, _ = nonmonotone_line_search(points, x, d, result)
        # gamma, x = armijo_line_search(x, np.asarray(d).reshape(-1), f, grad, gamma, points)
        f, grad = f_grad(x, points)

        ext_condition = (np.abs(x - x_old) > max_error).any()

        y = np.matrix(grad - grad_old).T
        s = np.matrix(x - x_old).T

        if it <= m:
            Y[:,it-1] = y.T
            S[:,it-1] = s.T
            rho[it-1] = 1 / (y.T * s).item()
        else:
            Y = np.roll(Y, -1)
            S = np.roll(S, -1)
            rho = np.roll(rho, -1)
            Y[:, m-1] = y
            S[:, m-1] = s
            rho[m-1] = 1 / (y.T * s).item()


    iter_time = time.time() - start_iters_time

    return np.array(result), time.time() - start_time, iter_time