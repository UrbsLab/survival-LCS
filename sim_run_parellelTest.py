import sys
import dask
from sim_utils import get_cluster, run_parellel

HPC = True

homedir = "/home/bandheyh/common/survival-LCS-telo"

print("Here 0")

cluster = get_cluster(output_path=homedir)

class Test:
    def run(self):
        print("Running")
        return 1

print("Here 1")

job_obj_list = [Test() for i in range(10)]

if HPC == True:
    delayed_results = []
    for model in job_obj_list:
        brier_df = dask.delayed(run_parellel)(model)
        delayed_results.append(brier_df)
    print("Here 2")
    results = dask.compute(*delayed_results)

cluster.close()

print(results)
