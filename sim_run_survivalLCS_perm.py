import os
import sys
import dask
import pickle
import shutil
import pandas as pd
from time import sleep
from dask.distributed import progress
from survivalLCSPermRun import ExperimentRun
from sim_utils import get_parameters, get_cluster, run_parellel

homedir = "/home/bandheyh/common/survival-LCS-telo"
sys.path.append(homedir)

HPC = True
DEBUG = False

if os.path.exists(homedir + '/dask_logs/'):
    shutil.rmtree(homedir + '/dask_logs/')
if not os.path.exists(homedir + '/dask_logs/'):
    os.mkdir(homedir + '/dask_logs/')

outputdir = homedir + "/pipeline"
model_list = ['me', 'epi', 'het', 'add']
nfeat_list = ['f100', 'f1000', 'f10000']
maf_list = ['maf0.2', 'maf0.4']
censor_list = [0.1, 0.4, 0.8]

time_label = "eventTime"
status_label = "eventStatus"
instance_label="inst"
T = 100
knots = 8

iterations = 200000
random_state = 42

print("Random Seed:", random_state)
cv_count = 5
pmethod = "random"
isContinuous = True
nu = 1
rulepop = 2000

n_perm = 5

if DEBUG:
    outputdir = homedir + "/test"
    model_list = ['me']
    censor_list = [ 0.1 ]
    nfeat_list = ['f100']
    maf_list = ['maf0.2']
    iterations = 1000
    cv_count = 3

### Create empty brier score DataFrame
brier_df = pd.DataFrame()

# make_folder_structure(outputdir, model_list)

job_obj_list = list()
brier_df_list = list()

for i in range(0,len(model_list)):
    for j in range(0,len(nfeat_list)):
        if nfeat_list[j] != 'f100': 
            continue
        for k in range(0,len(maf_list)):
            g, mtype, d, m, o, e, m0_path, m0_type, m1_path, m1_type = get_parameters(homedir, outputdir, 
                                                                                      model_list, nfeat_list, maf_list, 
                                                                                      i, j, k)
            gametes_data_path = g
            gametes_model_path_0 = m0_path
            gametes_model_path_1 = m1_path
            data_path = d
            model_path = m
            output_path = o
            experiment_name = e
            model0_type = m0_type
            model1_type = m1_type
            model_type = mtype

            for l in range(0, len(censor_list)):
                for m in range(0, cv_count):
                        for k in range(0, n_perm):
                            slcs = ExperimentRun(data_path, model_path, output_path, model_type, m, censor_list[l], k,
                                                iterations, nu, rulepop)
                            if HPC == False:
                                ibs = slcs.run()
                                brier_df_list.append(ibs)
                            else:
                                job_obj_list.append(slcs)

print("No of jobs:", len(job_obj_list))

if HPC == True:
    cluster = get_cluster(output_path=homedir)
    delayed_results = []
    for model in job_obj_list:
        brier_df = dask.delayed(run_parellel)(model)
        delayed_results.append(brier_df)
    results = dask.compute(*delayed_results)
    
    # print(print(cluster.scheduler_info()))

    sleep(10)
    while ((len(cluster.scheduler_info()["workers"]) > 0)):
        print("Running", len(cluster.scheduler_info()["workers"]), "workers")
        sleep(10)

    print("Errors:", sum(type(x) != pd.DataFrame for x in results))

    print(results)

    with open(outputdir + '/results_survivalLCS_perm_parallel.pkl', 'wb') as file:
        pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)
