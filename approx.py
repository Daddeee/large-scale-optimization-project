import numpy as np
import time

def crude_approximation(points, K):
    subset1 = np.random.choice(range(points.shape[0]), size=int(K), replace=False)    
    ind1 = np.zeros(points.shape[0], dtype=bool)
    ind1[subset1] = True
    pi = points[ind1]
    
    rest = ~ind1
    subset2 = np.random.choice(np.arange(points.shape[0])[rest], size=int(K), replace=False)    
    ind2 = np.zeros(points.shape[0], dtype=bool)
    ind2[subset2] = True
    pj = points[ind2]

    f = None
    for p in pi:
        norms = np.sum((p - pj)**2, axis=1)**0.5
        perc = np.percentile(norms, 65)
        if f is None or perc < f:
            f = perc

    return f


def approx(points, debug=False):
    start_time = time.time()
    max_error = 1e-5
    
    eps = 0.5

    T = 60/eps
    f = crude_approximation(points, T)
    
    return np.array([f]), time.time() - start_time