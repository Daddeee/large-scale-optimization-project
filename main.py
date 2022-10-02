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

class Instance:
    def __init__(self, points):
        self.points = points
        self.n = points.shape[1]
        self.m = points.shape[0]
        self.id = "{}-{}".format(self.n, self.m)

    def read(path):
        points = np.loadtxt(f)
        return Instance(points)

class ExperimentResult:
    def __init__(self, name, n, m, n_iter, time, iter_time, f):
        self.name = name
        self.n = n
        self.m = m
        self.n_iter = n_iter
        self.time = time
        self.iter_time = iter_time
        self.f = f

    def aggregate(other_results):
        assert len(other_results) > 0, "Empty results"
        name = other_results[0].name
        n = other_results[0].n
        m = other_results[0].m

        sum_n_iter, sum_time, sum_iter_time = 0, 0, 0
        for res in other_results:
            assert res.name == name
            assert res.n == n
            assert res.m == m
            sum_n_iter += res.n_iter
            sum_time += res.time
            sum_iter_time += res.iter_time
        
        n_iter = sum_n_iter / len(other_results)
        time = sum_time / len(other_results)
        iter_time = sum_iter_time / len(other_results)
        
        return ExperimentResult(name, n, m, n_iter, time, iter_time, 0)

    def to_row_solution(self):
        return {
            f"{self.name} obj": self.f, 
            f"{self.name} n iter": self.n_iter, 
            f"{self.name} time": self.time, 
            f"{self.name} iter time": self.iter_time
        }

    def to_row_solution_with_baseline(self, baseline):
        return {
            f"{self.name} obj": 100 * (self.f - baseline.f) / baseline.f, 
            f"{self.name} n iter": 100 * (self.n_iter - baseline.n_iter) / baseline.n_iter, 
            f"{self.name} time": 100 * (self.time - baseline.time) / baseline.time, 
            f"{self.name} iter time": 100 * (self.iter_time - baseline.iter_time) / baseline.iter_time
        }

    def to_row(self):
        h = self.to_row_solution()
        h['n'] = self.n
        h['m'] = self.m
        return h


class ExperimentManager:
    SOLVERS = {
        "weiszfeld": weiszfeld,
        "newton": newton,
        "bfgs": bfgs,
        # "modified": accelerated_weiszfeld
    }

    BASELINE = "weiszfeld"

    def __init__(self):
        self.results = {}

    def solve_all(self, instance):
        for name in self.SOLVERS:
            self.solve(instance, name)

    def solve(self, instance, name):
        print("Running {}".format(name))

        debug = False

        func = self.SOLVERS[name]

        y,t,t_iter = func(instance.points, debug)
        x = np.arange(0, y.shape[0])

        if not (instance.id in self.results):
            self.results[instance.id] = {}

        if not (name in self.results[instance.id]):
            self.results[instance.id][name] = []

        result = ExperimentResult(name, instance.n, instance.m, len(y), t, t_iter, np.amin(y))
        self.results[instance.id][name].append(result)

        print("obj={}, time={}, iterations={}".format(result.f, result.time, result.n_iter))


    def write_csv(self, outpath):
        rows = []
        for instance_id in self.results:
            res = self.results[instance_id]

            instance_results = []
            for solv in res:
                instance_results.append(ExperimentResult.aggregate(res[solv]))

            instance_results.sort(key=lambda x: x.name)

            baseline = None
            results = []
            for res in instance_results:
                if res.name == self.BASELINE:
                    baseline = res
                else:
                    results.append(res)
            
            h = None
            if baseline is not None:
                h = baseline.to_row()

            first = True
            for res in results:
                if first:
                    if h is None:
                        h = res.to_row()
                    first = False
                
                h = {**h, **res.to_row_solution()}

                # if baseline is None:
                #     h = {**h, **res.to_row_solution()}
                # else:
                #     h = {**h, **res.to_row_solution_with_baseline(baseline)}
            
            rows.append(h)

        df = pd.DataFrame(rows)
        df.to_csv(outpath, sep=";")


exp_manager = ExperimentManager()

result_directory = "results/"
instances_directory = "instances/"
for d in os.listdir(instances_directory):

    dim_dir = os.path.join(instances_directory, d)

    result_dim_dir = os.path.join(result_directory, d)
    if not os.path.exists(result_dim_dir):
        os.makedirs(result_dim_dir)

    for size in os.listdir(dim_dir):
        size_dir = os.path.join(dim_dir, size)

        instances = [os.path.join(size_dir, f) for f in os.listdir(size_dir) if ".txt" in f]

        for f in instances:
            instance = Instance.read(f)
            print("SOLVING {}".format(f))
            exp_manager.solve_all(instance)

        exp_manager.write_csv("result.csv")

