import numpy as np
import time
from common import get_start_point

def weiszfeld(points, debug=False):
    start_time = time.time()
    
    ap, is_optimal, f, d, t = get_start_point(points)
    if d is None:
        return np.array([f]), time.time() - start_time
    start_p = ap + t*d
    # x = np.average(points, axis=0)
    
    max_error = 1e-5
    ext_condition = True
    maxiters = 1000

    start_iters_time = time.time()
    result = []
    while ext_condition and len(result) < maxiters:
        sod = np.sum((points - start_p)**2, axis=1)**0.5
        new_p = np.sum((points.T/sod).T, axis=0) / sum(1/sod)
        ext_condition = (np.abs(new_p - start_p) > max_error).any()
        start_p = new_p
        f = sod.sum()
        result.append(f)

        if debug:
            print("f={} x={}".format(f, new_p))

    iter_time = time.time() - start_iters_time

    return np.array(result), time.time() - start_time, iter_time