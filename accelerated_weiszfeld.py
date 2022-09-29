import numpy as np
import time
from common import get_start_point

def accelerated_weiszfeld(points, debug=False):
    start_time = time.time()
    max_error = 1e-5
    ext_condition = True
    start_p, is_optimal, f, d, t = get_start_point(points)
    result = []
    while ext_condition:
        sod = np.sum((points - start_p)**2, axis=1)**0.5
        new_p = np.sum((points.T/sod).T, axis=0) / sum(1/sod)
        ext_condition = (np.abs(new_p - start_p) > max_error).any()
        start_p = new_p
        f = sod.sum()
        result.append(f)

        if debug:
            print("f={} x={}".format(f, new_p))

    return np.array(result), time.time() - start_time