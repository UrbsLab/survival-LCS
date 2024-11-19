import os
import time
import pickle
import numpy as np
import pandas as pd
import sksurv
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sim_utils import make_folder
from sLCS import survivalLCS

time_label = "eventTime"
status_label = "eventStatus"
instance_label="inst"
T = 100
random_state = 42

class ExperimentRun:

    def __init__(self, dpath, mpath, opath, mtype, cv, censor, perm, iterations=50000, nu=1, rulepop=1000):
        self.data_path = dpath
        self.model_path = mpath
        self.output_path = opath
        self.model_type = mtype
        self.cv = cv
        self.censor = censor
        self.iterations = iterations
        self.nu = nu
        self.rulepop = rulepop
        self.perm = perm
        
    def run(self):

        predList = None
        predProbs = None

        model_path_censor = self.model_path + '/cens_'+ str(self.censor)
        make_folder(model_path_censor)
        make_folder(model_path_censor+'/Perm/')

        output_path_censor = self.output_path + '/cens_'+ str(self.censor)
        make_folder(output_path_censor)
        make_folder(output_path_censor+'/Perm/')

        train_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor) + '_surv' + '_CV_'+str(self.cv)+'_Train.txt'
        data_train = pd.read_csv(train_file, sep='\t') #, header = 0
        instID = instance_label
        timeLabel = time_label
        censorLabel = status_label

        #Derive the attribute and phenotype array using the phenotype label
        dataFeatures_train = data_train.drop([timeLabel,censorLabel,instID],axis = 1).values
        dataEvents_train = data_train[[timeLabel,censorLabel]].values

        #Optional: Retrieve the headers for each attribute as a length n array
        dataHeaders_train = data_train.drop([timeLabel,censorLabel,instID],axis=1).columns.values

        #split dataEvents into two separate arrays (time and censoring)
        dataEventTimes_train = dataEvents_train[:,0]
        dataEventStatus_train = dataEvents_train[:,1]

        test_file = self.data_path + '/' + str(self.model_type) + '_cens'+ str(self.censor) + '_surv' + '_CV_'+str(self.cv)+'_Test.txt'
        data_test = pd.read_csv(test_file, sep='\t') #, headers = 0

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
        model = survivalLCS(learning_iterations=self.iterations, nu=self.nu, N=self.rulepop)
        trainedModel = model.fit(dataFeatures_train,dataEventTimes_train,dataEventStatus_train)

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
        if predList is None:
            predList = trainedModel.predict(dataFeatures_test)
        else:
            predList = np.append(predList, trainedModel.predict(dataFeatures_test))
        # print(predList)

        if predProbs is None:
            predProbs = pd.DataFrame(trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T
        else:
            predProbs = pd.concat([predProbs, pd.DataFrame(trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T])
        # print(predProbs.head())

        ### Pickle the model
        pickle.dump(trainedModel, open(self.model_path+'/cens_'+str(self.censor)+'/Perm/Perm_' + str(self.perm) +'_ExSTraCS_'+str(self.cv),'wb'))
        print("Pickled survivalLCS Model #"+str(self.cv))

        loopend = time.time()
        runtime = loopend - start
        # runtime_df.loc[len(runtime_df.index)] = [self.model_type,self.censor, self.output_path, dataFeatures_train.shape[1], runtime]
        # Obtain the integrated brier score
        try:
            times, b_scores = trainedModel.brier_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)

            col_name = 'b_scores_' + \
                str(os.path.basename(self.output_path)) + \
                    'perm' + str(self.perm) + \
                        '_cens' + str(self.censor) + \
                            '_cv' + str(self.cv)

            tb = pd.DataFrame({'times':times, col_name:b_scores})

            #sum, then average scores
            self.ibs_df = tb
            self.ibs_df.set_index('times',inplace=True)
            self.ibs_df.to_csv(self.output_path+'/cens_'+str(self.censor)+'/Perm/Perm_' + str(self.perm) + '_ExSTraCS_'+str(self.cv)+'_brierscores.csv')

            self.ibs_value = trainedModel.integrated_b_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)
            print("integrated_brier_score: ",self.ibs_value)
            with open(self.output_path+'/cens_'+str(self.censor)+'/Perm/Perm_' + str(self.perm) + '_ExSTraCS_'+str(self.cv)+'.txt','w') as file:
                file.write("Model, " + col_name + '\n')
                file.write("Integrated Brier Score, " + str(self.ibs_value))

        except Exception as e:
            print('Error generating integrated Brier scores', e)
            return e
        
        return self.ibs_df
    
    def get_output(self):
        return self.ibs_df
