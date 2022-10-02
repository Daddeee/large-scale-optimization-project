import os
from tqdm import tqdm
import numpy as np

directory = "instances/"

if not os.path.exists(directory):
    os.makedirs(directory)

max_coord = 100

np.random.seed(1337)

dims = [2, 3, 4, 5, 6, 7, 8, 9, 10]
sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
num_rand = 100

for dim in tqdm(dims):
    base_dir = "{}{}/".format(directory, dim)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for size in sizes:
        size_dir = "{}{}/".format(base_dir, size)
        if not os.path.exists(size_dir):
            os.makedirs(size_dir)

        for i in range(num_rand):
            points = np.random.uniform(low=0, high=max_coord, size=(size,dim))
            np.savetxt("{}{}.txt".format(size_dir, i), points)
