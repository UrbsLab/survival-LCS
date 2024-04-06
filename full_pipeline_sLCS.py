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
sys.path.append("/home/bandheyh/common/survival-lcs")


'''Sample Run Code
python full_pipeline_sLCS.py
python AnalysisPhase1.py --d ../Datasets/mp11_full.csv --o ../Outputs --e mp11 --inst Instance --group Group --iter 20000 --N 1000 --nu 10 --cluster 0

[rereqs
pip3 install pandas matplotlib scikit-learn scikit-ExSTraCS skrebate fastcluster seaborn networkx pygame pytest-shutil
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

#g = "/Users/alexaw/Documents/UrbsLab/full_pipeline/maineff_full_test_EDM-1/maineff_full_test_EDM-1_001.txt"
#m0_path = "/Users/alexaw/Documents/UrbsLab/full_pipeline/maineff_full_test_Models.txt"
#m0_type = "main_effect"
#mtype = "main_effect"
#d = "/Users/alexaw/Documents/UrbsLab/full_pipeline/cv_full_test"
#m = "/Users/alexaw/Documents/UrbsLab/full_pipeline/model_pickle_test"
#o = "/Users/alexaw/Documents/UrbsLab/full_pipeline/output_test"
#e =  "attempt1"
#s = True


g = "/Users/alexaw/Documents/UrbsLab/test/pipeline/merged_NBL_data_LCS_test.txt"
mtype = "main_effect" #or data_name
d = "/Users/alexaw/Documents/UrbsLab/test/pipeline/cv_full_test"
m = "/Users/alexaw/Documents/UrbsLab/test/pipeline/model_pickle_test"
o = "/Users/alexaw/Documents/UrbsLab/test/pipeline/output_test"
e =  "test_sim"
s = True


#other non-parameters
homedir = "/Users/alexaw/Documents/UrbsLab/full_pipeline/"


### Call the survival_LCS pipeline

from survival_LCS_pipeline import survivalLCS


### Run the survival LCS

survivalLCS = survivalLCS(g, mtype,d, m, o, e,iterations = 20000, cv = 5)

#### If the data is from GAMETES, create survival output
if s == True:
    survivalLCS.returnPenetrance()
    survivalLCS.returnSurvivalData()

else:
    os.rename(g,  homedir + str(mtype) + '_surv_' + str(date.today())) #else rename the file and copy it to the output dir
    shutil.copy(homedir + str(mtype) + '_surv_' + str(date.today()), o)


survivalLCS.returnCVDatasets()

survivalLCS.returnCVModelFiles()

survivalLCS.plot_results() #need to work on the plotting capabilities
