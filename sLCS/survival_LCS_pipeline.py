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
from survival_ExSTraCS import survival_ExSTraCS
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

class survivalLCS():
    def __init__(self,g,mtype,d,m,o,e, brier_df,cox_brier_df,m0_path = None, m0_type = None,m1_path =None,m1_type =None,T = 100,k = 8,c = [0.1],time_label = "eventTime",status_label = "eventStatus",instance_label="inst",cv = 5,pmethod = "random",random_state = None,isContinuous = True, iterations =20000, nu = 1, rp = 1000, cluster=1,m1=2,m2=3):


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
            os.mkdir(self.data_path)

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)



    #------------------------------------------------------------------------------------------
    # parse the GAMETES model files - this probably shouldnt be a function?
    #------------------------------------------------------------------------------------------
    #importGametes(self,gametesData,gametesModel0,model0,model1,full_model,gametesModel1=None)
    def returnPenetrance(self):  #parseModelFile(self,gametesModel0, gametesModel1,model0, model1,full_model)
        import_Gametes = importGametes(self.gametes_data_path,self.gametes_model_path_0, self.model0_type,self.model1_type,self.model_type,self.gametes_model_path_1)
        self.P, self.P0, self.P1 = import_Gametes.parseModelFile(self.gametes_model_path_0,self.gametes_model_path_1,self.model0_type,self.model1_type,self.model_type)
        X = import_Gametes.gametesData

    #------------------------------------------------------------------------------------------
    # Create survival dataset
    #------------------------------------------------------------------------------------------

    def returnSurvivalData(self):
        import_Gametes = importGametes(self.gametes_data_path,self.gametes_model_path_0, self.model0_type,self.model1_type,self.model_type,self.gametes_model_path_1)
        test_genSurvSim = genSurvSim(import_Gametes, self.T, self.model_type)
        #baseline = test_genSurvSim.baseline_build(self.T, self.knots)
        for i in range(0,len(self.censor)): #updated to produce a dataset for each censoring proportion
            survData,survDists = test_genSurvSim.generate_time(test_genSurvSim.X, import_Gametes.names, self.P, self.P0, self.P1, self.model_type,self.censor[i]) #names is from importGametes, will this import properly?

            #weird error in heterogeneous datasets, added extra column "Unnamed...", this removes that
            survData = survData.loc[:, ~survData.columns.str.startswith('Unnamed')]
            #need to save data to one of the output locations
            survData.to_csv(self.output_path+'/' + str(self.model_type) + '_cens'+ str(self.censor[i]) + '_surv_'+ str(date.today()), index = False, sep = '\t')



    #------------------------------------------------------------------------------------------
    # Plot true event times
    #------------------------------------------------------------------------------------------

    def plotTrueEventTimes(self, data):
        data = pd.read_csv(data, sep = '\t', header = 0)
        #print(list(data.columns))
        dataEventTimes = list(data[self.time_label])


        plt.figure(figsize=(6, 4))
        plt.xlim([0,max(dataEventTimes)])
        plt.xlabel('True Event Times', fontsize=14)
        plt.ylabel('# of Instances', fontsize=14)
        plt.hist(x=dataEventTimes, bins='auto', color='cadetblue', alpha=0.7, rwidth=0.85)
        plt.savefig(self.output_path+'/true_hist'+self.model_type+'.png')
        plt.close()



    #------------------------------------------------------------------------------------------
    # Create CV datasets
    #------------------------------------------------------------------------------------------
    def returnCVDatasets(self): #note the naming of the files will be different depending on whether simulated or real data is used.
        if self.gametes_model_path_0 is not None:
            for i in range(0, len(self.censor)):
                cv_part = cvPartitioner(self.output_path+'/' + str(self.model_type) +'_cens'+str(self.censor[i])+'_surv_'+ str(date.today()),self.data_path, self.cv_count,self.pmethod,self.time_label)
        else:
            for i in range(0,len(self.censor)):
                cv_part = cvPartitioner(self.output_path+'/' + str(self.model_type)+'_surv_'+ str(date.today()),self.data_path, self.cv_count,self.pmethod,self.time_label)

    #------------------------------------------------------------------------------------------
    # Run survival-LCS on simulated CV datasets, pickle model files for LCS_DIVE
    #------------------------------------------------------------------------------------------
    def returnCVModelFiles(self):
        import matplotlib.pyplot as plt

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
                ### Train the survival_ExSTraCS model
                model = survival_ExSTraCS(learning_iterations = self.iterations,nu=self.nu,N=self.rulepop)
                self.trainedModel = model.fit(dataFeatures_train,dataEventTimes_train,dataEventStatus_train)



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
                    rsf = make_pipeline(RandomSurvivalForest(random_state=42))
                    rsf.fit(dataFeatures_train,scoreEvents_train)

                    perm_rsf = PermutationImportance(rsf.steps[-1][1], n_iter=10, random_state=42).fit(dataFeatures_train,scoreEvents_train)
                    rsf_data = perm_rsf.results_
                    rsf_data = pd.DataFrame(rsf_data, columns=dataHeaders_train)

                    #concat dfs
                    self.cox_df = pd.concat([self.cox_df, cox_data], ignore_index=True, axis=0)
                    self.rsf_df = pd.concat([self.rsf_df, rsf_data], ignore_index=True, axis=0)
                else:
                    print("Comparison approaches not run on datasets with > 100 features")


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

                ### Pickle the model
                pickle.dump(self.trainedModel, open(self.model_path+'/cens_'+str(self.censor[j])+'/ExSTraCS_'+str(i)+'_'+str(date.today()),'wb'))
                #model.pickle_model(defaultExportDir+'/ExSTraCS_'+str(i))
                print("Pickled survivalLCS Model #"+str(i))

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


                    #sum, then average scores
                    ibs_df = pd.concat([tb,ibs_df],axis=1,sort=False).reset_index()
                    ibs_df.set_index('times',inplace=True)
                except Exception as e:
                    print('Error generating integrated Brier scores', e)
                    pass


                #self.trainedModel.plot_ibs(times, b_scores)
                #plt.savefig(self.output_path+'/ibs_plot_'+self.model_type+'cv_'+str(i)+'.svg')
                #plt.close()


       	    #PLOT predicted times - need to also intake max time to set X axis
            #trainedModel.plotPreds(predList) #plots the predicted times of the test data across all CVs
            #plt.savefig(self.output_path+'/pred_hist'+self.model_type+'.svg')
            #plt.close()

            #Output the individual survival distributions - fix this to call from trainedModel
            #ax = predProbs.plot(legend = False, title = 'Individual Survival Probabilities ('+ self.model_type+' model)')
            #ax.set_ylabel("Survival Probability")
            #ax.set_xlabel("Time")
            #plt.savefig(self.output_path+'/pred_survival_probs_'+self.model_type+'.svg')
            #plt.close()

            #for ibs plotting, generate columns mean and CI
            ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])] = ibs_df.mean(axis = 1)
            #print('ibs_df :', ibs_df)
            ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])+'_ci_lower'] = ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])] - (ibs_df.std(axis = 1)*2)
            ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])+'_ci_upper'] = ibs_df[str(os.path.basename(self.output_path)) + '_cens'+ str(self.censor[j])] + (ibs_df.std(axis = 1)*2)
            #ibs_df.to_csv(self.output_path+'/ibs_data_'+self.model_type+'.txt', index = False)
            print('ibs_df :', ibs_df)

            self.brier_df = pd.concat([self.brier_df,ibs_df], axis = 1, sort = False).reset_index()
            self.brier_df.set_index('times', inplace = True)
            self.brier_df = self.brier_df.loc[:, ~self.brier_df.columns.str.startswith('b_scores')]
            print('self.brier_df from one dataset (across 3 cvs): ', self.brier_df)


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

            #Plot the true event times of each model
            self.plotTrueEventTimes(self.output_path+'/' + str(self.model_type) + '_cens'+ str(self.censor[j]) + '_surv_' + str(date.today()))


            ## Plot results from EACH model
            try:
                self.plot_results(self.censor[j])
            except Exception as ex:
                print("Plot results error: ",ex)

    #------------------------------------------------------------------------------------------
    # Run survival-LCS on REAL WORLD CV datasets, pickle model files for LCS_DIVE
    #------------------------------------------------------------------------------------------
    def returnCVModelFilesReal(self):
        import matplotlib.pyplot as plt

        #data_path = '/Users/alexaw/Documents/UrbsLab/skExSTraCS/cv_data_5fold' #this needs to move
        instID = self.instance_label
        # iterate over files in
        # that directory
        train_dfs = []
        test_dfs = []
        self.brier_df = pd.DataFrame() #this way, self.brier_df will reset each time returnCVModelFiles is called
        self.brier_df['times'] = range(self.T + 1)
        self.brier_df.set_index('times',inplace=True)
        #self.model_path_censor = None
        self.dataFeatures = None
        self.dataEvents = None
        self.dataHeaders = None
        self.predList = None
        self.predProbs = None
        self.attSums = None
        self.featImp = None
        self.ibs_avg = None

        ibs_df = pd.DataFrame() #hopefully this resets the ibs_df after every set of CVs.
        ibs_df['times'] = range(self.T + 1)
        ibs_df.set_index('times',inplace=True)
        
        for i in range(self.cv_count):

            dataset_train = None
            dataset_test = None


            train_file = self.data_path+ '/' + str(self.model_type) + '_surv_' + str(date.today()) + '_CV_'+str(i)+'_Train.txt'
            data_train = pd.read_csv(train_file, sep='\t') #, header = 0
            #data_train = data_train.fillna(np.nan)
            cols=[i for i in data_train.columns if i not in ["inst"]]
            for col in cols:
                data_train[col]=pd.to_numeric(data_train[col], errors='coerce')

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


            test_file = self.data_path+ '/' + str(self.model_type) + '_surv_' + str(date.today()) + '_CV_'+str(i)+'_Test.txt'
            data_test = pd.read_csv(test_file, sep='\t') #, headers = 0
            #data_test = data_test.fillna(np.nan)
            cols=[i for i in data_test.columns if i not in ["inst"]]
            for col in cols:
                data_test[col]=pd.to_numeric(data_test[col],errors='coerce')
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
            ### Train the survival_ExSTraCS model
            model = survival_ExSTraCS(learning_iterations = self.iterations,nu=self.nu,N=self.rulepop)
            self.trainedModel = model.fit(dataFeatures_train,dataEventTimes_train,dataEventStatus_train)



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
            if dataFeatures_train.shape[1] < 101 and np.isnan(np.sum(dataFeatures_train)) == False:
                CoxPH = make_pipeline(CoxPHSurvivalAnalysis(alpha = 0.00001))
                CoxPH.fit(dataFeatures_train, scoreEvents_train)
                perm = PermutationImportance(CoxPH.steps[-1][1], n_iter=10, random_state=42).fit(dataFeatures_train,scoreEvents_train)

                cox_data = perm.results_
                cox_data = pd.DataFrame(cox_data, columns=dataHeaders_train) #remove index label

                #Random survival forest
                rsf = make_pipeline(RandomSurvivalForest(random_state=42))
                rsf.fit(dataFeatures_train,scoreEvents_train)

                perm_rsf = PermutationImportance(rsf.steps[-1][1], n_iter=10, random_state=42).fit(dataFeatures_train,scoreEvents_train)
                rsf_data = perm_rsf.results_
                rsf_data = pd.DataFrame(rsf_data, columns=dataHeaders_train)

                #concat dfs
                self.cox_df = pd.concat([self.cox_df, cox_data], ignore_index=True, axis=0)
                self.rsf_df = pd.concat([self.rsf_df, rsf_data], ignore_index=True, axis=0)
            else:
                print("Comparison approaches not run on datasets with > 100 features")


            # -------------------------------------------------------------------------------------------
            ### Survival Prediction - LCS
            #--------------------------------------------------------------------------------------------
            ##HERE Make this function also generate all of the relevant graphs and save them in the appropriate output file
            if self.predList is None:
                self.predList = self.trainedModel.predict(dataFeatures_test)
            else:
                self.predList = np.append(self.predList, self.trainedModel.predict(dataFeatures_test))
            print('predList: ',self.predList)
            
            #Per sharon, having this individual level data would be useful
            #indiv_data_preds = pd.concat([pd.DataFrame(self.predList),pd.DataFrame(dataEvents_test)], axis = 1)
            #print(indiv_data_preds.head())
            #indiv_data_preds.to_csv(self.output_path+'/indv_data_preds_'+str(i)+'_'+str(date.today())+'.txt', index = False)

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

            ### Pickle the model
            pickle.dump(self.trainedModel, open(self.model_path+'/ExSTraCS_'+str(i)+'_'+str(date.today()),'wb'))
            #model.pickle_model(defaultExportDir+'/ExSTraCS_'+str(i))
            print("Pickled survivalLCS Model #"+str(i))

            loopend = time.time()
            runtime = loopend - start
            self.runtime_df_real.loc[len(self.runtime_df.index)] = [self.model_type, self.output_path, dataFeatures_train.shape[1], runtime]

                #Obtain the integrated brier score
            try:
                times, b_scores = self.trainedModel.brier_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)

                tb =pd.DataFrame({'times':times, 'b_scores'+str(i):b_scores})
                tb.set_index('times',inplace=True)


                #sum, then average scores
                ibs_df = pd.concat([tb,ibs_df],axis=1,sort=False).reset_index()
                ibs_df.set_index('times',inplace=True)
            except:
                print('Error generating integrated Brier scores')
                pass


        

        #for ibs plotting, generate columns mean and CI
        ibs_df[str(os.path.basename(self.output_path))] = ibs_df.mean(axis = 1)
        #print('ibs_df :', ibs_df)
        ibs_df[str(os.path.basename(self.output_path)) + '_ci_lower'] = ibs_df[str(os.path.basename(self.output_path))] - (ibs_df.std(axis = 1)*2)
        ibs_df[str(os.path.basename(self.output_path)) + '_ci_upper'] = ibs_df[str(os.path.basename(self.output_path))] + (ibs_df.std(axis = 1)*2)
        #ibs_df.to_csv(self.output_path+'/ibs_data_'+self.model_type+'.txt', index = False)
        print('ibs_df :', ibs_df)

        self.brier_df = pd.concat([self.brier_df,ibs_df], axis = 1, sort = False).reset_index()
        self.brier_df.set_index('times', inplace = True)
        self.brier_df = self.brier_df.loc[:, ~self.brier_df.columns.str.startswith('b_scores')]
        print('self.brier_df from one dataset (across 3 cvs): ', self.brier_df)


        #for KM plots of top 5 features
        self.dataFeatures = np.append(dataFeatures_train, dataFeatures_test, axis = 0)
        self.dataEvents = np.append(dataEvents_train, dataEvents_test, axis = 0)
        self.dataHeaders = dataHeaders_train

        #Plot the true event times of each model    
        self.plotTrueEventTimes(self.output_path+'/' +'NBL' + '_surv_' + str(date.today()))

        ## Plot results from EACH model
        try:
            self.plot_results_real()
        except Exception as ex:
            print(ex)           

    #------------------------------------------------------------------------------------------
    # Return brier scores
    #------------------------------------------------------------------------------------------

    def returnIBSresults(self):
        return self.brier_df

    #------------------------------------------------------------------------------------------
    # Return brier scores
    #------------------------------------------------------------------------------------------

    def return_cox_IBSresults(self):
        return self.cox_bscore_df

    #------------------------------------------------------------------------------------------
    # Run plot results of merged CV datasets - SIMULATED data
    #------------------------------------------------------------------------------------------

    def plot_results(self, cens_prop):
        #PLOT predicted times as a histogram
        self.trainedModel.plotPreds(self.predList) #plots the predicted times of the test data across all CVs
        plt.savefig(self.output_path+'/cens_'+str(cens_prop)+'/pred_hist'+self.model_type+'.png')
        plt.close()

        #Output the individual survival distributions - fix this to call from trainedModel
        ax = self.predProbs.plot(legend = False, title = 'Individual Survival Probabilities ('+ self.model_type+' model)')
        ax.set_ylabel("Survival Probability")
        ax.set_xlabel("Time")
        plt.savefig(self.output_path+'/cens_'+str(cens_prop)+'/pred_survival_probs_'+self.model_type+'.png')
        plt.close()

        #k = 5 #make this a default param?
        #fix this to make figures save.
        self.trainedModel.plotKM(self.output_path,self.dataFeatures, self.dataEvents, self.dataHeaders, k=5, cens_prop = cens_prop, attSums = self.attSums)
        #plt.savefig(self.output_path+'/top_feat_KM_'+self.model_type+'.png')
        #plt.close()

        #Plot sLCS feature importances (boxplot)
        self.trainedModel.plotfeatImp(self.output_path, self.model_type, self.dataFeatures, self.dataHeaders, self.featImp, cens_prop)

        if self.cox_df is not None:
            #Plot cox model results - feature importance (permutation)
            cox_meds = self.cox_df.median()
            cox_meds = cox_meds.sort_values(ascending=False)
            self.cox_df = self.cox_df[cox_meds.index]
            self.cox_df = self.cox_df.iloc[:,:11] #plot top 10 features

            cox_fig, ax = plt.subplots(figsize=(10,7))
            self.cox_df.boxplot(ax=ax)
            ax.set_title('CoxPH Feature Importances')
            plt.savefig(self.output_path+'/cens_'+str(cens_prop)+'/top_feat_CoxPH_'+self.model_type+'.png')
            plt.close()

            #Plot RSF model results - feature importance (permutation)
            rsf_meds = self.rsf_df.median()
            rsf_meds = rsf_meds.sort_values(ascending=False)
            self.rsf_df = self.rsf_df[rsf_meds.index]
            self.rsf_df = self.rsf_df.iloc[:,:11]

            rsf_fig, ax = plt.subplots(figsize=(10,7))
            self.rsf_df.boxplot(ax=ax)
            ax.set_title('RSF Feature Importances')
            plt.savefig(self.output_path+'/cens_'+str(cens_prop)+'/top_feat_RSF_'+self.model_type+'.png')
            plt.close()
        else:
            print('No comparison plots generated')

    #------------------------------------------------------------------------------------------
    # Run plot results of merged CV datasets - REAL data
    #------------------------------------------------------------------------------------------

    def plot_results_real(self):
        #PLOT predicted times as a histogram
        self.trainedModel.plotPreds(self.predList) #plots the predicted times of the test data across all CVs
        plt.savefig(self.output_path+'/pred_hist'+self.model_type+'.png')
        plt.close()

        #Output the individual survival distributions - fix this to call from trainedModel
        ax = self.predProbs.plot(legend = False, title = 'Individual Survival Probabilities ('+ self.model_type+' model)')
        ax.set_ylabel("Survival Probability")
        ax.set_xlabel("Time")
        plt.savefig(self.output_path+'/pred_survival_probs_'+self.model_type+'.png')
        plt.close()

        #k = 5 #make this a default param?
        #fix this to make figures save.
        self.trainedModel.plotKM(self.output_path,self.dataFeatures, self.dataEvents, self.dataHeaders, k=5, attSums = self.attSums)
        #plt.savefig(self.output_path+'/top_feat_KM_'+self.model_type+'.png')
        #plt.close()

        #Plot sLCS feature importances (boxplot)
        self.trainedModel.plotfeatImp(self.output_path, self.model_type, self.dataFeatures, self.dataHeaders, self.featImp)

        if self.cox_df is not None:
            #Plot cox model results - feature importance (permutation)
            cox_meds = self.cox_df.median()
            cox_meds = cox_meds.sort_values(ascending=False)
            self.cox_df = self.cox_df[cox_meds.index]
            self.cox_df = self.cox_df.iloc[:,:11] #plot top 10 features

            cox_fig, ax = plt.subplots(figsize=(10,7))
            self.cox_df.boxplot(ax=ax)
            ax.set_title('CoxPH Feature Importances')
            plt.savefig(self.output_path+'/top_feat_CoxPH_'+self.model_type+'.png')
            plt.close()

            #Plot RSF model results - feature importance (permutation)
            rsf_meds = self.rsf_df.median()
            rsf_meds = rsf_meds.sort_values(ascending=False)
            self.rsf_df = self.rsf_df[rsf_meds.index]
            self.rsf_df = self.rsf_df.iloc[:,:11]

            rsf_fig, ax = plt.subplots(figsize=(10,7))
            self.rsf_df.boxplot(ax=ax)
            ax.set_title('RSF Feature Importances')
            plt.savefig(self.output_path+'/top_feat_RSF_'+self.model_type+'.png')
            plt.close()
        else:
            print('No comparison plots generated')



    #------------------------------------------------------------------------------------------
    # Run plot results of merged CV datasets
    #------------------------------------------------------------------------------------------

    def plot_ibs(self,brier_df):
        plt.figure(figsize=(10, 10))
        #pyplot.vlines(empDist, 0, 0.05, linestyles ="solid", colors ="k")
        plt.xlabel('Time')
        plt.ylabel('Brier score')

        plt.plot(times, b_scores)




    #------------------------------------------------------------------------------------------
    # Run LCS_DIVE & produce visualizations
    #------------------------------------------------------------------------------------------

    def dive_phase1(self):
        cmd1 = 'python3 AnalysisPhase1_pretrained.py ' +'--d '+ self.data_path + ' --m '+ self.model_path + ' --o ' + self.output_path+'/DIVE_results'+ ' --e '+ self.experiment_name
        #print('pythonpath: ',os.environ['PYTHONPATH'])
        print(cmd1)
        my_env = os.environ.copy()
        my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
        subprocess.run(cmd1, shell = True, env = my_env, check = True)
        #AnalysisPhase1 = AnalysisPhase1_pretrained()
        #AnalysisPhase1.main(self.data_path,self.model_path,self.output_path,self.experiment_name)

        cmd2 = 'python3 AnalysisPhase2.py '+'--o '+ self.output_path+'/DIVE_results'+' --e '+self.experiment_name+' --c '+ str(0)
        print(cmd2)
        subprocess.run(cmd2,shell = True,env=my_env, check = True)


    #------------------------------------------------------------------------------------------
    # Submit job to the cluster ??? use this here or nah?
    #------------------------------------------------------------------------------------------

    def submitJob(uniquejobname,scratchpath,logpath,runpath,dataset,outfile,algorithm,discthresh,outcomelabel,neighbors):
        """ Submit Job to the cluster. """
        jobName = scratchpath+'/'+uniquejobname+'_'+str(time.time())+'_run.sh'
        shFile = open(jobName, 'w')
        shFile.write('#!/bin/bash\n')
        shFile.write('#BSUB -J '+uniquejobname+'_'+str(time.time())+'\n')
        shFile.write('#BSUB -o ' + logpath+'/'+uniquejobname+'.o\n')
        shFile.write('#BSUB -e ' + logpath+'/'+uniquejobname+'.e\n\n')
        shFile.write('python '+runpath+'/'+'run_scikit-rebate.py '+str(dataset)+' '+str(outfile)+' '+str(algorithm)+' '+str(discthresh)+' '+str(outcomelabel)+' '+str(neighbors)+'\n')
        shFile.close()
        os.system('bsub < '+jobName)
