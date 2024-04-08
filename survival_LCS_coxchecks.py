#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import pandas as pd
import os
import copy
import subprocess
from datetime import date
import time
import sksurv
import subprocess
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
from sksurv.ensemble import RandomSurvivalForest
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from survival_Timer import Timer
from survival_OfflineEnvironment import OfflineEnvironment
from survival_ExpertKnowledge import ExpertKnowledge
from survival_AttributeTracking import AttributeTracking
from survival_ClassifierSet import ClassifierSet
from survival_Prediction import Prediction
from survival_RuleCompaction import RuleCompaction
from survival_IterationRecord import IterationRecord
from survival_Pareto import Pareto
from survival_Metrics import Metrics
from survival_ExSTraCS import ExSTraCS

class survivalLCS_coxChecks():
    def __init__(self,g,mtype,d,m,o,e, brier_df,cox_brier_df,m0_path = None, m0_type = None,m1_path =None,m1_type =None,T = 100,k = 8,c = [0.1],time_label = "eventTime",status_label = "eventStatus",instance_label="inst",cv = 5,pmethod = "random",random_state = None,isContinuous = True, iterations =50000, nu = 1, rp = 1000, cluster=1,m1=2,m2=3):


        ''' #Datasets for simulation
        parser.add_argument('--g', dest='gametes_data_path', type=str, help='path to directory containing GAMETES data file or QCed genomic survival data')
        parser.add_argument('--m0_path', dest='gametes_model_path_0', type=str, help='path to directory XXX', defaut = None)
        parser.add_argument('--m1_path', dest='gametes_model_path_1', type=str, help='path to directory XXX', default = None)
        parser.add_argument('--m0type', dest='model0_type', type=str, help='Model0 type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"', default = "None")
        parser.add_argument('--m1type', dest='model1_type', type=str, help='Model1 type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"', default = None)
        parser.add_argument('--mtype', dest='model_type', type=str, help='Full model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"', default = "main_effect")
        parser.add_argument('--t', dest='T', type=int, help='time', default = 100)
        parser.add_argument('--k', dest='knots', type=int, help='number of knots for baseline survival', default = 8)
        parser.add_argument('--c', dest='censor', type = list, help = 'proportion to censor', default = [0.1])
        #Datasets for survival-LCS
        parser.add_argument('--d', dest='data_path', type=str, help='path to directory containing presplit train/test datasets ending with _CV_Test/Train.csv')
        parser.add_argument('--m', dest='model_path', type=str, help='path to directory containing pretrained ExSTraCS Models labeled ExStraCS_CV')
        parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
        parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')
        #update to time and status instead of just "class"
        parser.add_argument('--time', dest='time_label', type=str, default="eventTime")
        parser.add_argument('--status',dest='status_label', type=str, default="eventStatus")
        parser.add_argument('--inst', dest='instance_label', type=str, default="inst") #need to have this, creates a mess if IDs are none
        parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=5)
        parser.add_argument('--p', dest='pmethod', type=str, help='Specify the partitioning approach from the following options (random, stratified, matched)', default = 'random')
        parser.add_argument('--random-state',dest='random_state',type=str,default=None)
        parser.add_argument('--cont',dest = 'isContinuous', help='Boolean: Specify if the endpoint is continuous-valued.', action='store_true')
        #LCS hyperparameters
        parser.add_argument('--iter', dest = 'iterations', type=int, help='number of iterations to run the survival-LCS', default=20000)
        parser.add_argument('--nu',dest= 'nu', type=int, default = 1)
        parser.add_argument('--rp', dest='rulepop', type=int, help='size of rule population',default = 1000)
        #Cluster parameters
        parser.add_argument('--cluster', dest='do_cluster', type=int, default=1)
        parser.add_argument('--m1', dest='memory1', type=int, default=2)
        parser.add_argument('--m2', dest='memory2', type=int, default=3)
        '''

        self.gametes_data_path = g
        self.gametes_model_path_0 = m0_path
        self.gametes_model_path_1 = m1_path
        self.data_path = d
        self.model_path = m
        self.output_path = o
        self.experiment_name = e
        self.model0_type = m0_type
        self.model1_type = m1_type
        self.model_type = mtype #add parameter with name of original dataset


        self.time_label = time_label
        self.status_label = status_label
        self.instance_label = instance_label
        self.T = T
        self.knots = k
        self.censor = c

        self.iterations = iterations
        self.random_state = random_state

        print(self.random_state)
        self.cv_count = cv
        self.pmethod = pmethod
        self.isContinuous = isContinuous
        self.nu = nu
        self.rulepop = rp

        self.brier_df = brier_df
        self.brier_df['times'] = range(self.T + 1)
        self.brier_df.set_index('times',inplace=True)

        self.cox_bscore_df = cox_brier_df
        self.cox_bscore_df['times'] = range(self.T + 1)
        self.cox_bscore_df.set_index('times',inplace=True)


        #for comparison approaches
        self.cox_df = pd.DataFrame()
        self.rsf_df = pd.DataFrame()

        #for timing each loop
        self.runtime_df = pd.DataFrame(columns = ["Model Type", "Censoring", "MAF","Nfeat","Time"])
        self.runtime_df_real = pd.DataFrame(columns = ["Model Type", "MAF","Nfeat","Time"])

        #XXX
        self.P = None
        self.P0 = None
        self.P1 = None
        self.ibs_avg = None

        if self.random_state == None:
            self.random_state = random.randint(0, 1000000)
        else:
            self.random_state = int(self.random_state)
        self.do_cluster = cluster
        self.memory1 = m1
        self.memory2 = m2


        # Create experiment folders and check path validity
        #need path to
        if not os.path.exists(self.gametes_data_path):
            print("gametes data path is: ", self.gametes_data_path)
            raise Exception("Provided gametes_data_path does not exist")
        #if self.gametes_model_path_0 is not None:
            #if not os.path.exists(self.gametes_model_path_0):
                #raise Exception("Provided gametes_model_path_0 does not exist")
        if self.gametes_model_path_1 is not None:
            if not os.path.exists(self.gametes_model_path_1):
                raise Exception("Provided gametes_model_path_1 does not exist")

        for char in self.experiment_name:
            if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_':
                raise Exception('Experiment Name must be alphanumeric')

        if not os.path.exists(self.data_path):
            print("data path is ",self.data_path)
            os.mkdir(self.data_path)

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    #------------------------------------------------------------------------------------------
    # Run Cox and PI analysis on simulated CV datasets
    #------------------------------------------------------------------------------------------
    def returnCVModelFiles(self):

        instID = self.instance_label
        train_dfs = []
        test_dfs = []
        self.brier_df = pd.DataFrame() #this way, self.brier_df will reset each time returnCVModelFiles is called
        self.brier_df['times'] = range(self.T + 1)
        self.brier_df.set_index('times',inplace=True)
        self.cox_bscore_df = pd.DataFrame()
        self.cox_bscore_df['times'] = range(self.T + 1)
        self.cox_bscore_df.set_index('times',inplace=True)
        self.model_path_censor = None
        self.dataFeatures = None
        self.dataEvents = None
        self.dataHeaders = None
        self.predList = None
        self.predProbs = None
        self.attSums = None
        self.featImp = None
        self.ibs_avg = None
        for j in range(0,len(self.censor)):
            ibs_df = pd.DataFrame() #hopefully this resets the ibs_df after every set of CVs.
            ibs_df['times'] = range(self.T + 1)
            ibs_df.set_index('times',inplace=True)
            #same for cox ibs data
            cb_df = pd.DataFrame() #hopefully this resets the ibs_df after every set of CVs.
            cb_df['times'] = range(self.T + 1)
            cb_df.set_index('times',inplace=True)
            for i in range(self.cv_count):
                dataset_train = None
                dataset_test = None

                self.model_path_censor = self.model_path + '/cens_'+ str(self.censor[j])
                if not os.path.exists(self.model_path_censor):
                    os.mkdir(self.model_path_censor)

                self.output_path_censor = self.output_path + '/cens_'+ str(self.censor[j])
                if not os.path.exists(self.output_path_censor):
                    os.mkdir(self.output_path_censor)

                train_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor[j]) + '_surv_'+ str(date.today()) + '_CV_'+str(i)+'_Train.txt'
                data_train = pd.read_csv(train_file, sep='\t') #, header = 0
                timeLabel = self.time_label
                censorLabel = self.status_label

                #Derive the attribute and phenotype array using the phenotype label
                dataFeatures_train = data_train.drop([timeLabel,censorLabel,instID],axis = 1).values
                dataEvents_train = data_train[[timeLabel,censorLabel]].values

                #Optional: Retrieve the headers for each attribute as a length n array
                dataHeaders_train = data_train.drop([timeLabel,censorLabel,instID],axis=1).columns.values

                #split dataEvents into two separate arrays (time and censoring)
                dataEventTimes_train = dataEvents_train[:,0]
                dataEventStatus_train = dataEvents_train[:,1]


                test_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor[j]) + '_surv_'+ str(date.today()) + '_CV_'+str(i)+'_Test.txt'
                data_test = pd.read_csv(test_file, sep='\t') #, headers = 0
                timeLabel = 'eventTime'
                censorLabel = 'eventStatus'

                #Derive the attribute and phenotype array using the phenotype label
                dataFeatures_test = data_test.drop([timeLabel,censorLabel,instID],axis = 1).values
                dataEvents_test = data_test[[timeLabel,censorLabel]].values

                #Optional: Retrieve the headers for each attribute as a length n array
                dataHeaders_test = data_test.drop([timeLabel,censorLabel,instID],axis=1).columns.values

                #split dataEvents into two separate arrays (time and censoring)
                dataEventTimes_test = dataEvents_test[:,0]
                dataEventStatus_test = dataEvents_test[:,1]


                start = time.time()

                #Format data for brier score
                #dataEvents_train[:, [1, 0]] = dataEvents_train[:, [0, 1]]
                #dataEvents_test[:, [1, 0]] = dataEvents_test[:, [0, 1]]

                scoreEvents_train = np.flip(dataEvents_train, 1)
                scoreEvents_test = np.flip(dataEvents_test, 1)

                scoreEvents_train = np.core.records.fromarrays(scoreEvents_train.transpose(),names='cens, time', formats = '?, <f8')
                scoreEvents_test = np.core.records.fromarrays(scoreEvents_test.transpose(),names='cens, time', formats = '?, <f8')


                ### Convert float data to int
                dataEventTimes_train = dataEventTimes_train.astype('int64')
                dataEventTimes_test = dataEventTimes_test.astype('int64')
                dataEventStatus_train = dataEventStatus_train.astype('int64')
                dataEventStatus_test = dataEventStatus_test.astype('int64')

                # -------------------------------------------------------------------------------------------
                ### Run other sklearn survival analyses
                #--------------------------------------------------------------------------------------------


                # Cox Proportional Hazards Model - maybe run this only if features <= 100
                if dataFeatures_train.shape[1] < 101:
                    CoxPH = make_pipeline(CoxPHSurvivalAnalysis(alpha = 0.00001))
                    est =  CoxPH.fit(dataFeatures_train, scoreEvents_train)

                    survs = est.predict_survival_function(dataFeatures_test)#use test data
                    #print('max event times test: ',max(dataEventTimes_test))
                    #print('max event times train: ', max(dataEventTimes_train))
                    cox_times = np.arange(max(min(dataEventTimes_test), min(dataEventTimes_train)), min(max(dataEventTimes_test),max(dataEventTimes_train))) #recall that these values must be within the range of the test times
                    #print(cox_times)
                    preds = np.asarray([[fn(t) for t in cox_times] for fn in survs])
                    #print("cox Preds: ", preds)

                    try:
                        times, cox_bscores = sksurv.metrics.brier_score(scoreEvents_test, scoreEvents_test, preds, cox_times)

                        cb =pd.DataFrame({'times':times, 'b_scores'+str(i):cox_bscores})
                        cb.set_index('times',inplace=True)

                        cb_df = pd.concat([cb,cb_df],axis=1,sort=False).reset_index()
                        cb_df.set_index('times',inplace=True)

                    except Exception as e:
                        print(e, 'No Cox brier scores generated')

                    perm = PermutationImportance(CoxPH.steps[-1][1], n_iter=10, random_state=42).fit(dataFeatures_train,scoreEvents_train)

                    cox_data = perm.results_
                    cox_data = pd.DataFrame(cox_data, columns=dataHeaders_train) #remove index label

                    #Random survival forest
                    # rsf = make_pipeline(RandomSurvivalForest(random_state=42))
                    # rsf.fit(dataFeatures_train,scoreEvents_train)

                    # perm_rsf = PermutationImportance(rsf.steps[-1][1], n_iter=10, random_state=42).fit(dataFeatures_train,scoreEvents_train)
                    # rsf_data = perm_rsf.results_
                    # rsf_data = pd.DataFrame(rsf_data, columns=dataHeaders_train)

                    #concat dfs
                    self.cox_df = pd.concat([self.cox_df, cox_data], ignore_index=True, axis=0)
                    self.perm_cox_df = self.cox_df
                    # self.rsf_df = pd.concat([self.rsf_df, rsf_data], ignore_index=True, axis=0)
                else:
                    print("Comparison approaches not run on datasets with > 100 features")

            try:
                cb_df[str(os.path.basename(self.output_path))] = cb_df.mean(axis = 1)
                cb_df[str(os.path.basename(self.output_path)) + '_ci_lower'] = cb_df[str(os.path.basename(self.output_path))] - (cb_df.std(axis = 1)*2)
                cb_df[str(os.path.basename(self.output_path)) + '_ci_upper'] = cb_df[str(os.path.basename(self.output_path))] + (cb_df.std(axis = 1)*2)

                self.cox_bscore_df = pd.concat([self.cox_bscore_df,cb_df], axis = 1, sort = False).reset_index()
                self.cox_bscore_df.set_index('times', inplace = True)
                self.cox_bscore_df = self.cox_bscore_df.loc[:, ~self.cox_bscore_df.columns.str.startswith('b_scores')]
            except Exception as e:
                print(e)
                continue

            #for KM plots of top 5 features
            self.dataFeatures = np.append(dataFeatures_train, dataFeatures_test, axis = 0)
            self.dataEvents = np.append(dataEvents_train, dataEvents_test, axis = 0)
            self.dataHeaders = dataHeaders_train
            

    #------------------------------------------------------------------------------------------
    # Return brier scores
    #------------------------------------------------------------------------------------------

    def return_cox_IBSresults(self):
        return self.cox_bscore_df
