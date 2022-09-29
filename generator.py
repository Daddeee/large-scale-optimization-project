import os
import numpy as np

directory = "instances/"

if not os.path.exists(directory):
    os.makedirs(directory)

max_coord = 200000

dims = [2, 5, 10, 20]
sizes = [1000, 2000, 5000, 10000, 50000, 100000]

for dim in dims:
    base_dir = "{}{}/".format(directory, dim)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for size in sizes:
        points = np.random.uniform(low=0, high=max_coord, size=(size,dim))
        np.savetxt("{}{}.txt".format(base_dir, size), points)
