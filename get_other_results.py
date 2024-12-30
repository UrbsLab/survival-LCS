import os
import sys
import dask
import pickle
import shutil
import pandas as pd
from time import sleep
from survivalLCSOtherOutputRun import ExperimentRun
from sim_utils import get_parameters, get_cluster, run_parellel

homedir = "/home/bandheyh/common/survival-LCS-telo"
sys.path.append(homedir)

HPC = False
DEBUG = False

if os.path.exists(homedir + '/dask_logs/'):
    shutil.rmtree(homedir + '/dask_logs/')
if not os.path.exists(homedir + '/dask_logs/'):
    os.mkdir(homedir + '/dask_logs/')

outputdir = homedir + "/pipeline_randomspline"
model_list = ['me', 'epi', 'het', 'add']
nfeat_list = ['f100', 'f1000', 'f10000']
maf_list = ['maf0.2', 'maf0.4']
censor_list = [0.1, 0.4, 0.8]

time_label = "eventTime"
status_label = "eventStatus"
instance_label = "inst"
random_state = 42

print("Random Seed:", random_state)
cv_count = 5

if DEBUG:
    outputdir = homedir + "/test"
    model_list = ['me']
    censor_list = [ 0.1 ]
    nfeat_list = ['f100']
    maf_list = ['maf0.2']
    cv_count = 3

# make_folder_structure(outputdir, model_list)

job_obj_list = list()
num = 0

for i in range(0,len(model_list)):
    for j in range(0,len(nfeat_list)):
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
                slcs = ExperimentRun(data_path, model_path, output_path, model_type, cv_count, censor_list[l])
                if HPC == False:
                    try:
                        slcs.run()
                        num+=1
                        print(num, "Items Run")
                    except:
                        num+=1
                        print("Items", num, "Failed")
                else:
                    job_obj_list.append(slcs)

if HPC == True:
    print("No of jobs:", len(job_obj_list))
    cluster = get_cluster(output_path=homedir)
    delayed_results = []
    for model in job_obj_list:
        brier_df = dask.delayed(run_parellel)(model)
        delayed_results.append(brier_df)
    results = dask.compute(*delayed_results)
    
    # cluster.close()
    print(print(cluster.scheduler_info()))

    while ((len(cluster.scheduler_info()["workers"]) > 0)):
        # print("Running", len(cluster.scheduler_info()["workers"]), "workers")
        sleep(60)

    print("Finished")
