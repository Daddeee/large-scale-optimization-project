import numpy as np

def nonmonotone_line_search(x, d, hist, M=10, sig1=0.1, sig2=0.9, gam=1e-4, maxiter=1000):
    alpha = 1
    f_max = max(obj_hist[-M:])
    f_prev = f(x)
    for i in range(maxiter):
        f_val = f(x + alpha * d)
        g = grad(x)
        gtd = np.dot(g, d)
        if f_val <= f_max + gam * alpha * gtd:
            return alpha, x + alpha * d, f_val, g
        a1 = -0.5 * (alpha**2) * gtd / (f_val - f_prev - alpha * gtd)
        if sig1 <= a1 and a1 <= sig2 * alpha:
            alpha = a1
        else:
            alpha /= 2
        
        f_prev = f_val

    return None, None, None, None

def f_grad_hess(x, points):
    diffs = x - points
    hess_second_term_numerator = np.matmul(np.expand_dims(diffs, axis=2), np.expand_dims(diffs, axis=1))
    idents = np.stack([np.identity(points.shape[1])]*points.shape[0], axis=0).shape
    sod = np.sum(diffs**2, axis=1)**0.5
    f = np.sum(sod)
    grad = np.sum(diffs / sod[:,None], axis=0)
    import pdb; pdb.set_trace()
    hess = idents / sod[:, None, None] - hess_second_term_numerator / (sod**3)[:, None, None]
    return f, grad, hess

def euclidean_trick(x):
    """Euclidean square distance matrix.
    
    Inputs:
    x: (N, m) numpy array
    y: (N, m) numpy array
    
    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = (x_ij - y_ij)^2
    """
    t = np.einsum('ij,ij->i', x, x)
    x2 = t[:, np.newaxis]
    y2 = t[np.newaxis, :]

    xy = x @ x.T

    return np.abs(x2 + y2 - 2. * xy)**0.5


def get_start_point(points):
    min_f, min_s, min_p, min_i = None, None, None, None
    dists = euclidean_trick(points)
    sods = np.sum(dists, axis=0)
    min_i = np.argmin(sods)
    min_s = dists[min_i]
    if np.isnan(min_s).any():
        print("aaaaaaaa")
        import pdb; pdb.set_trace()
    min_f = sods[min_i]
    min_p = points[min_i]
    r = (points - min_p) / min_s[:,None]
    r[min_i] = np.zeros(len(min_p))
    min_r = np.sum(r)
    min_r_norm = np.sum(min_r**2)**0.5
    is_optimal = min_r_norm <= 1
    if is_optimal:
        return min_p, min_f, True, None, None
    else:
        d = - min_r / min_r_norm
        t = (min_r_norm - 1) / np.sum(1 / min_s)
        return min_p + d*t, None, False, d, t

def get_min_anchor_direction_and_stepsize(points):
    min_f, min_s, min_p, min_i = None, None, None, None
    dists = euclidean_trick(points)
    sods = np.sum(dists, axis=0)
    min_i = np.argmin(sods)
    min_s = dists[min_i]
    min_f = sods[min_i]
    min_p = points[min_i]

    norm_s = np.linalg.norm(min_s)
    d = -min_s / norm_s
    l = np.sum(1 / min_s[:,None])
    t = (norm_s - 1) / l

    return min_p, d, t

# def f_grad_hess(x, points):
#     diffs = x - points
#     so2d = np.sum(diffs**2, axis=1)
#     sod = so2d**0.5
#     so3d = sod**3
#     f = np.sum(sod)
#     grad = np.sum(diffs / sod[:,None], axis=0)

#     hess = np.zeros((len(x),len(x)))
#     for k, p in enumerate(points):
#         h = so2d[k] * np.identity(len(x)) - np.transpose([diffs[k]]) * diffs[k]
#         h = h / so3d[k]
#         hess += h

#     return f, grad, hess

def f_grad(x, points):
    diffs = x - points
    so2d = np.sum(diffs**2, axis=1)
    sod = so2d**0.5
    f = np.sum(sod)
    grad = np.sum(diffs / sod[:,None], axis=0)
    return f, grad

def f(x, points):
    diffs = x - points
    return np.sum(np.sum(diffs**2, axis=1)**0.5)


def armijo_line_search(x0, d, fx0, dfx0, gamma, points):
    beta = 0.5
    sigma = 1e-1
    maxit = 1000
    x = x0
    for it in range(0,maxit):
        x = x0 + gamma * d
        fx = np.sum(np.sum((points - x)**2, axis=1)**0.5)
        if fx <= fx0 + gamma * sigma * np.dot(dfx0, d):
            break
        gamma = beta*gamma

    return gamma, x