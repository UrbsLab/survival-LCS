# %% [markdown]
# # SurvivalLCS Experiment Runs

# %% [markdown]
# ## Import and Setup

# %% [markdown]
# ### Load packages

# %%
import os
import pandas as pd
import numpy as np
import random
import sys
import glob
from datetime import date
import argparse
from random import shuffle
from random import sample
import matplotlib.pyplot as plt
import sys
import shutil
import sksurv
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from survival_LCS_pipeline import survivalLCS

# %%
sys.path.append("/home/bandheyh/common/survival-lcs")

# %%
plt.ioff()
plt.ioff()

# %% [markdown]
# ### Import py scripts

# %%
import survival_AttributeTracking
import survival_Classifier
import survival_ClassifierSet
import survival_DataManagement
import survival_ExpertKnowledge
import survival_ExSTraCS
import survival_IterationRecord
import survival_Pareto
import survival_Prediction
import survival_StringEnumerator
import survival_OfflineEnvironment
import survival_Timer
import survival_RuleCompaction
import survival_Metrics
import utils
import nonparametric_estimators
import importGametes
import survival_data_simulator

# %% [markdown]
# ### Test run scripts interactively

# %%
%run -i survival_AttributeTracking.py
%run -i survival_Classifier.py
%run -i survival_ClassifierSet.py
%run -i survival_DataManagement.py
%run -i survival_ExpertKnowledge.py
%run -i survival_ExSTraCS.py
%run -i survival_IterationRecord.py
%run -i survival_Pareto.py
%run -i survival_Prediction.py
%run -i survival_StringEnumerator.py
%run -i survival_OfflineEnvironment.py
%run -i survival_Timer.py
%run -i survival_RuleCompaction.py
%run -i survival_Metrics.py
%run -i utils.py
%run -i nonparametric_estimators.py

# %% [markdown]
# ## Survival-LCS Parameters
# 
# ### Set file names and necessary parameters

# %%
# parameter to run using hpc resources
HPC = True

homedir = "/home/bandheyh/common/survival-lcs/pipeline"
models = ['me', 'epi', 'het', 'add']
m0s = []

c = [0.1,0.4,0.8]
nfeat = ['f100','f1000', 'f10000'] #add f10000 when on cluster
maf = ['maf0.2','maf0.4']

iterations = 50000
cv_splits = 5

DEBUG = False
if DEBUG:
    models = ['me']
    c = [0.1]
    nfeat = ['f100', 'f1000']
    maf = ['maf0.2', 'maf0.4']
    iterations = 500
    cv_splits = 3

### Create empty brier score DataFrame
brier_df = pd.DataFrame()
cox_brier_df = pd.DataFrame()

# other non-parameters

simulated = True # CHANGE THIS TO FALSE IF RUNNING REAL DATA

lcs_run = True
dtype_list = []

# %% [markdown]
# ### Import the survival_LCS pipeline

# %%
from survival_LCS_pipeline import survivalLCS

# %% [markdown]
# ### Making the directory structure

# %% [markdown]
# You'll need to create the following folders and subfolders, INSIDE of the home directory for output files:
# 1. cv_sim_data (with subfolders: cv_me, cv_epi, cv_het, cv_add)
# 2. pickled_cv_models (with subfolders: me, epi, het, add)
# 3. sim_lcs_output (with subfolders: me, epi, het, add)

# %%
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

# %%
def make_folder_structure(homedir, models, overwrite=True):
    if overwrite==True:
        make_folder(homedir+'/cv_sim_data/')
        make_folder(homedir+'/pickled_cv_models/')
        make_folder(homedir+'/sim_lcs_output/')
        for model in models:
            make_folder(homedir+'/cv_sim_data/cv_' + model)
            make_folder(homedir+'/pickled_cv_models/' + model)
            make_folder(homedir+'/sim_lcs_output/' + model)
    else:
        raise NotImplemented

# %%
make_folder_structure(homedir, models)

# %% [markdown]
# ### Run the survival_LCS pipeline

# %%
def get_parameters(models, nfeat, maf, i, j, k):

    g = homedir + '/' + 'simulated_datasets/' + \
        'EDM-1_one_of_each/'+str(models[i]) + \
        '_' + str(nfeat[j]) + '_' + str(maf[k]) + '_' + 'EDM-1_01.txt'
    dtype = str(models[i]) + '_' + str(nfeat[j]) + '_' + str(maf[k])
    dtype_list.append(dtype)
    print(g)

    d = homedir + '/' + 'cv_sim_data/cv_' + str(models[i]) + '/' + dtype
    m = homedir + '/' + 'pickled_cv_models/' + str(models[i]) + '/' + dtype
    o = homedir + '/' + 'sim_lcs_output/' + str(models[i]) + '/' + dtype

    ### Set m0_path
    if models[i] in ['me','add','het']:
        m0_path = homedir+'/'+'simulated_datasets/'+'EDM-1_one_of_each/model_files/me_h0.2_'+str(maf[k])+'_Models.txt'
    else:
        m0_path = homedir+'/'+'simulated_datasets/'+'EDM-1_one_of_each/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'

    ### Set m1_path
    if models[i] in ['me','epi']:
        m1_path = None
    else:
        m1_path = homedir+'/'+'simulated_datasets/'+'EDM-1_one_of_each/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'

    ### Set m0_type
    if models[i] in ['me','add','het']:
        m0_type = 'main_effect'
    else:
        m0_type = '2way_epistasis'

    ### Set m1_type
    if models[i] in ['me', 'epi']:
        m1_type = None
    else:
        m1_type = '2way_epistasis'

    ### Set mtype
    if models[i] == 'me':
        mtype = 'main_effect'
    elif models[i] == 'epi':
        mtype = '2way_epistasis'
    elif models[i] == 'add':
        mtype = 'additive'
    else:
        mtype = 'heterogeneous'


    e = "testallsims"
    print(str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k]))

    return g, mtype, d, m, o, e,brier_df,cox_brier_df, m0_path, m0_type, m1_path, m1_type



# %%
def run_slcs(survivalLCS):
    survivalLCS.returnPenetrance()
    survivalLCS.returnSurvivalData()

    if lcs_run == True:
        survivalLCS.returnCVDatasets()
        survivalLCS.returnCVModelFiles()

        current_ibs = survivalLCS.returnIBSresults()
        current_ibs = current_ibs.rename(columns={"mean": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k]), 
                                                  "ci_lower": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])+'_ci_lower', 
                                                  "ci_upper": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])+'_ci_upper'})
    else:
        print("Datasets generated only")

    print(survivalLCS.model_type)

    return current_ibs

# %%
def make_breir_output(brier_df_list, output_path, model_type, models, dtype_list, i):
    brier_df = pd.concat(brier_df_list, axis = 1, sort = False).reset_index()

    brier_df.to_csv(homedir +'/'+'sim_lcs_output/'+str(models[i])+'/ibs_data_'+mtype+'.txt', index = False)

    plt.figure(figsize=(10, 10))
    plt.xlabel('Time')
    plt.ylabel('Brier score')
    plt.ylim(0,1)

    for i in range(1,len(dtype_list)):
        plt.plot(brier['times'], brier[dtype_list[i]],label = brier[dtype_list[i]].name)
        plt.fill_between(brier['times'], brier[dtype_list[i]+'_ci_lower'], brier[dtype_list[i]+'_ci_upper'], color='b', alpha=.1)
    plt.savefig(output_path+'/brier_scores_'+model_type + '.png')

# %%
%%capture
from survival_LCS_pipeline import survivalLCS
job_obj_list = list()
for i in range(0,len(models)):
    for j in range(0,len(nfeat)):
        brier_df_list = list()
        for k in range(0,len(maf)):
            g, mtype, d, m, o, e,brier_df,cox_brier_df, m0_path, m0_type, m1_path, m1_type = get_parameters(models, nfeat, maf, i, j, k)
            survivalLCS = survivalLCS(g, mtype, d, m, o, e,brier_df,cox_brier_df, m0_path, m0_type, m1_path, m1_type, 
                                      c = c,iterations = iterations, cv = cv_splits)
            if HPC == False:
                current_ibs = run_slcs(survivalLCS)
                brier_df_list.append(current_ibs)
            else:
                job_obj_list.append(survivalLCS)
        if HPC == False:
            if lcs_run == True:
                make_breir_output(brier_df_list, survivalLCS.output_path, survivalLCS.model_type, models, dtype_list, i)
            else:
                print('LCS not run, no brier scores available')

# %% [markdown]
# ## HPC Code

# %%
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster, LSFCluster, SGECluster

# %%
def get_cluster(cluster_type='SLURM', output_path=".", queue='defq', memory=4):
    client = None
    try:
        if cluster_type == 'SLURM':
            cluster = SLURMCluster(queue=queue,
                                   cores=1,
                                   memory=str(memory) + "G",
                                   walltime="24:00:00",
                                   log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == "LSF":
            cluster = LSFCluster(queue=queue,
                                 cores=1,
                                 mem=memory * 1000000000,
                                 memory=str(memory) + "G",
                                 walltime="24:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == 'UGE':
            cluster = SGECluster(queue=queue,
                                 cores=1,
                                 memory=str(memory) + "G",
                                 resource_spec="mem_free=" + str(memory) + "G",
                                 walltime="24:00:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=400)
        elif cluster_type == 'Local':
            c = Client()
            cluster = c.cluster
        else:
            raise Exception("Unknown or Unsupported Cluster Type")
        client = Client(cluster)
    except Exception as e:
        print(e)
        raise Exception("Exception: Unknown Exception")
    print("Running dask-cluster")
    print(client.scheduler_info())
    return client

# %%
cluster = get_cluster()

# %%
make_folder('./dask_logs/')

# %%
def run_parallel(model):
    brier_df = run_slcs(model)
    return brier_df

# %%
if HPC:
    results = dask.compute([dask.delayed(run_parallel)(model) for model in job_obj_list])


