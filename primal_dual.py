import numpy as np
import time

def primal_dual(points, debug=False):
    start_time = time.time()
    max_error = 1e-5
    ext_condition = True

    x = np.average(points, axis=0)
    q = np.zeros(points.shape)
    M = len(points)

    result = []
    while ext_condition:
        prev_x = x

        to_proj = x - points + q

        to_proj_norm = np.linalg.norm(to_proj, axis=1)
        to_proj_norm[to_proj_norm < 1] = 1

        p = to_proj / to_proj_norm[:,None]

        y = x + q - p

        x = (1/M) * np.sum(y, axis=0)

        q = p - (1/M) * np.sum(p) 

        sod = np.sum((points - x)**2, axis=1)**0.5
        ext_condition = (np.abs(x - prev_x) > max_error).any()
        
        f = sod.sum()
        result.append(f)

        if debug:
            print("f={} x={}".format(f, x))

    return np.array(result), time.time() - start_time