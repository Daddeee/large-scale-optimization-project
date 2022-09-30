import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from weiszfeld import weiszfeld
from newton import newton
from primal_dual import primal_dual
from bfgs import bfgs
from modified_weiszfeld import modified_weiszfeld
from accelerated_weiszfeld import accelerated_weiszfeld


def void_experiment(func, name, color, iter_list, time_list):
    iter_list.append(0)
    time_list.append(0)

def run_experiment(func, name, color, iter_list, time_list):
    print("Running {}".format(name))

    debug = False

    y,t = func(points, debug)
    x = np.arange(0, y.shape[0])
    plt.plot(x, y, color=color, label=name)

    iter_list.append(len(y))
    time_list.append(t)

    print("obj={}, time={}, iterations={}".format(np.amin(y), t, len(y)))



result_directory = "results/"
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

n_list = []
m_list = []
weiszfeld_time_list = []
weiszfeld_iter_list = []
newton_time_list = []
newton_iter_list = []
bfgs_time_list = []
bfgs_iter_list = []
primal_dual_time_list = []
primal_dual_iter_list = []

instances_directory = "instances/"
for d in os.listdir(instances_directory):

    n = int(d)

    instance_dir = os.path.join(instances_directory, d)

    result_dir = os.path.join(result_directory, d)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    instances = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if ".txt" in f]

    for f in instances:
        points = np.loadtxt(f)

        m = points.shape[0]

        n_list.append(n)
        m_list.append(m)

        print("SOLVING {}".format(f))

        run_experiment(weiszfeld, "weiszfeld", "red", weiszfeld_iter_list, weiszfeld_time_list)
        run_experiment(newton, "newton", "blue", newton_iter_list, newton_time_list)
        run_experiment(bfgs, "bfgs", "green", bfgs_iter_list, bfgs_time_list)
        run_experiment(accelerated_weiszfeld, "modified", "yellow", primal_dual_iter_list, primal_dual_time_list)

        plt.legend()

        filename = "{}.png".format(points.shape[0])

        plt.savefig(os.path.join(result_dir, filename))
        plt.clf()


df = pd.DataFrame({
    'n': n_list,
    'm': m_list,
    'weiszfeld iter': weiszfeld_iter_list,
    'weiszfeld time': weiszfeld_time_list,
    'newton iter': newton_iter_list,
    'newton time': newton_time_list,
    'bfgs iter': bfgs_iter_list,
    'bfgs time': bfgs_time_list,
    'primal-dual iter': primal_dual_iter_list,
    'primal-dual time': primal_dual_time_list
})

df.to_csv("result.csv", sep=";")