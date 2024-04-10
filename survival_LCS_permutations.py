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



from importGametes import *
from survival_data_simulator import *
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
from AnalysisPhase1_pretrained import *
from AnalysisPhase2 import *
from NetworkVisualization import *
from cvPartitioner import *

#Need to write a job script for this
#from XXX import full_job


'''Sample Run Code:
python AnalysisPhase1_pretrained.py --o /Users/robert/Desktop/outputs/test1/mp6/viz-outputs --e root --d /Users/robert/Desktop/outputs/test1/mp6/CVDatasets --m /Users/robert/Desktop/outputs/test1/mp6/training/pickledModels --inst Instance --cv 3 --cluster 0
:param gametes: XXX
:param T:       Max time
:param model:   Must be string. Model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"
:param knots:   Number of knots to generate baseline survival model, default = 8
'''

class survivalLCS_permutations():
    def __init__(self,g,mtype,d,m,o,e, brier_df,cox_brier_df,m0_path = None, m0_type = None,m1_path =None,m1_type =None,T = 100,k = 8,c = [0.1],time_label = "eventTime",status_label = "eventStatus",instance_label="inst",cv = 5,pmethod = "random",random_state = None,isContinuous = True, iterations =50000, nu = 1, rp = 1000, cluster=1,m1=2,m2=3, perm_n=20):


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
        self.perm_n = perm_n


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


    def returnCVModelFiles(self):
        # import matplotlib.pyplot as plt

        #data_path = '/Users/alexaw/Documents/UrbsLab/skExSTraCS/cv_data_5fold' #this needs to move
        instID = self.instance_label
        # iterate over files in
        # that directory
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
            for k in range(self.perm_n):
                for i in range(self.cv_count):
                    dataset_train = None
                    dataset_test = None

                    self.model_path_censor = self.model_path + '/cens_'+ str(self.censor[j])
                    if not os.path.exists(self.model_path_censor):
                        os.mkdir(self.model_path_censor)

                    self.output_path_censor = self.output_path + '/cens_'+ str(self.censor[j])
                    if not os.path.exists(self.output_path_censor):
                        os.mkdir(self.output_path_censor)

                    train_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor[j]) + '_surv_'+ str('2024-04-07') + '_CV_'+str(i)+'_Train.txt'
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


                    test_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor[j]) + '_surv_'+ str('2024-04-07') + '_CV_'+str(i)+'_Test.txt'
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

                    np.random.shuffle(dataEventTimes_train)

                    start = time.time()
                    ### Train the survival_ExSTraCS model
                    model = ExSTraCS(learning_iterations = self.iterations,nu=self.nu,N=self.rulepop)
                    self.trainedModel = model.fit(dataFeatures_train,dataEventTimes_train,dataEventStatus_train)

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
                    ### Survival Prediction - LCS
                    #--------------------------------------------------------------------------------------------
                    ##HERE Make this function also generate all of the relevant graphs and save them in the appropriate output file
                    if self.predList is None:
                        self.predList = self.trainedModel.predict(dataFeatures_test)
                    else:
                        self.predList = np.append(self.predList, self.trainedModel.predict(dataFeatures_test))

                    #print(self.predList)

                    if self.predProbs is None:
                        self.predProbs = pd.DataFrame(self.trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T
                        #print(predProbs.head())
                    else:
                        self.predProbs = pd.concat([self.predProbs, pd.DataFrame(self.trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T])

                    #(add i += 1?)

                    #Retrieve attsums, (average across all CV or just use sum across??)
                    if self.attSums is None:
                        self.attSums = np.array(self.trainedModel.AT.getSumGlobalAttTrack(self.trainedModel))
                        self.featImp = self.attSums #this is going to be an nd.array used to make a box plot later
                    else:
                        self.attSums = self.attSums + np.array(self.trainedModel.AT.getSumGlobalAttTrack(self.trainedModel))
                        self.featImp = np.vstack((self.featImp, np.array(self.trainedModel.AT.getSumGlobalAttTrack(self.trainedModel))))
                #     print("featImp: ", self.featImp)
                #     if i == self.cv_count
                #     attSums / cv_count

                    loopend = time.time()
                    runtime = loopend - start
                    self.runtime_df.loc[len(self.runtime_df.index)] = [self.model_type,self.censor[j], self.output_path, dataFeatures_train.shape[1], runtime]

                    #Obtain the integrated brier score
                    try:
                        times, b_scores = self.trainedModel.brier_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)

                        #ibs_value = self.trainedModel.integrated_b_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)
                        #print("integrated_brier_score: ",ibs_value)

                        tb = pd.DataFrame({'times':times, 'b_scores'+str(i):b_scores})
                        tb.set_index('times',inplace=True)

                        # #sum, then average scores
                        # ibs_df = pd.concat([tb,ibs_df],axis=1,sort=False).reset_index()
                        # ibs_df.set_index('times',inplace=True)

                                        #for ibs plotting, generate columns mean and CI
                        ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j]) + '_perm' + str(k) + '_cv' + str(i)] = tb.mean(axis = 1)
                        #print('ibs_df :', ibs_df)
                        # ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j]) + '_perm' + str(k) + '_ci_lower'] = ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])] - (ibs_df.std(axis = 1)*2)
                        # ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j]) + '_perm' + str(k) + '_ci_upper'] = ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])] + (ibs_df.std(axis = 1)*2)
                        #ibs_df.to_csv(self.output_path+'/ibs_data_'+self.model_type+'.txt', index = False)
                        print('ibs_df :', ibs_df)
                    except Exception as e:
                        print('Error generating integrated Brier scores', e)
                        pass


                    #self.trainedModel.plot_ibs(times, b_scores)
                    #plt.savefig(self.output_path+'/ibs_plot_'+self.model_type+'cv_'+str(i)+'.svg')
                    #plt.close()

                self.brier_df = pd.concat([self.brier_df,ibs_df], axis = 1, sort = False).reset_index()
                self.brier_df.set_index('times', inplace = True)
                self.brier_df = self.brier_df.loc[:, ~self.brier_df.columns.str.startswith('b_scores')]
                print('self.brier_df from one dataset (across 3 cvs): ', self.brier_df)

    #------------------------------------------------------------------------------------------
    # Return brier scores
    #------------------------------------------------------------------------------------------

    def returnIBSresults(self):
        return self.brier_df
    