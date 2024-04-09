#!/usr/bin/env python
# coding: utf-8

### Load packages
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

sys.path.append("/home/bandheyh/common/survival-lcs")


'''Sample Run Code
python3 sim_full_pipeline_sLCS.py
python AnalysisPhase1.py --d ../Datasets/mp11_full.csv --o ../Outputs --e mp11 --inst Instance --group Group --iter 20000 --N 1000 --nu 10 --cluster 0

[rereqs
pip install pandas matplotlib scikit-learn skrebate fastcluster seaborn networkx pygame pytest-shutil eli5 scikit-survival
'''

### Import py scripts
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

### Survival-LCS Parameters



### Set file names and necessary parameters

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
    nfeat = ['f100']
    maf = ['maf0.2']
    iterations = 500
    cv_splits = 3

### Create empty brier score DataFrame
brier_df = pd.DataFrame()
cox_brier_df = pd.DataFrame()

#other non-parameters

simulated = True #CHANGE THIS TO FALSE IF RUNNING REAL DATA

lcs_run = True
dtype_list = []

def make_folder_structure(homedir, models, overwrite=True):
    def make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
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

make_folder_structure(homedir, models)


### Call the survival_LCS pipeline
from survival_LCS_pipeline import survivalLCS


for i in range(0,len(models)):
    for j in range(0,len(nfeat)):
        for k in range(0,len(maf)):
            g = homedir+'/'+'simulated_datasets/'+ 'EDM-1_one_of_each/'+str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])+'_'+'EDM-1_01.txt'
            print(g)
            dtype = str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])
            dtype_list.append(dtype)
            d = homedir +'/'+'cv_sim_data/cv_'+str(models[i])+'/'+dtype
            m = homedir +'/'+'pickled_cv_models/'+str(models[i])+'/'+dtype
            o = homedir +'/'+'sim_lcs_output/'+str(models[i])+'/'+dtype

            ### Set m0_path
            if models[i] in ['me','add','het']:
                m0_path = homedir+'/'+'simulated_datasets/'+ 'EDM-1_one_of_each/model_files/me_h0.2_'+str(maf[k])+'_Models.txt'
            else:
                m0_path = homedir+'/'+'simulated_datasets/'+ 'EDM-1_one_of_each/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'
            ### Set m1_path
            if models[i] in ['me','epi']:
                m1_path = None
            else:
                m1_path = homedir+'/'+'simulated_datasets/'+ 'EDM-1_one_of_each/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'

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

            ### INITIALIZE the sLCS script
            from survival_LCS_pipeline import survivalLCS
            survivalLCS = survivalLCS(g, mtype,d, m, o, e,brier_df,cox_brier_df, m0_path, m0_type, m1_path,m1_type,
                                      c = c,iterations = iterations, cv = cv_splits)

            ### Using data from GAMETES, create survival output
            survivalLCS.returnPenetrance()
            survivalLCS.returnSurvivalData()

            if lcs_run == True:
                survivalLCS.returnCVDatasets()

                survivalLCS.returnCVModelFiles()


                ###tentative new function for sensitivity analysis
                current_ibs = survivalLCS.returnIBSresults()
                current_ibs = current_ibs.rename(columns={"mean": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k]), "ci_lower": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])+'_ci_lower', "ci_upper": str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k])+'_ci_upper'})
                brier_df = pd.concat([brier_df,current_ibs], axis = 1, sort = False).reset_index()
                #print('brier_df:', brier_df)
                brier_df.to_csv(o+'/ibs_data_'+dtype+'.txt', index = False)

                ## Changed the line below, now called within survivalLCS.returnCVModelFiles()
                #survivalLCS.plot_results() #need to work on the plotting capabilities, input brier_df
            else:
                print("Datasets generated only")

    if lcs_run == True:
        ### Save the brier_df for each set of MODELS
        brier_df.to_csv(homedir +'/'+'sim_lcs_output/'+str(models[i])+'/ibs_data_'+mtype+'.txt', index = False)

        plt.figure(figsize=(10, 10))
        #pyplot.vlines(empDist, 0, 0.05, linestyles ="solid", colors ="k")
        plt.xlabel('Time')
        plt.ylabel('Brier score')
        plt.ylim(0,1)

        for i in range(1,len(dtype_list)):
            plt.plot(brier['times'], brier[dtype_list[i]],label = brier[dtype_list[i]].name)
            plt.fill_between(brier['times'], brier[dtype_list[i]+'_ci_lower'], brier[dtype_list[i]+'_ci_upper'], color='b', alpha=.1)
        plt.savefig(survivalLCS.output_path+'/brier_scores_'+survivalLCS.model_type + '.png')
    else:
        print('LCS not run, no brier scores available')