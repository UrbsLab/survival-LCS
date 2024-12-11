import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sLCS import survivalLCS
sns.set_theme()

time_label = "eventTime"
status_label = "eventStatus"
instance_label="inst"
random_state = 42

class ExperimentRun:

    def __init__(self, dpath, mpath, opath, mtype, cv_count, censor):
        self.data_path = dpath
        self.model_path = mpath
        self.output_path = opath
        self.model_type = mtype
        self.cv_count = cv_count
        self.censor = censor
        
    def run(self):

        self.dataFeatures = None
        self.dataEvents = None
        self.dataHeaders = None
        self.predList = None
        self.predProbs = None
        self.attSums = None
        self.featImp = None

        for i in range(self.cv_count):

            self.model_path_censor = self.model_path + '/cens_'+ str(self.censor)
            if not os.path.exists(self.model_path_censor):
                os.mkdir(self.model_path_censor)

            self.output_path_censor = self.output_path + '/cens_'+ str(self.censor)
            if not os.path.exists(self.output_path_censor):
                os.mkdir(self.output_path_censor)

            train_file = self.data_path+ '/' + str(self.model_type) + '_cens'+ str(self.censor) + '_surv' + '_CV_'+str(i)+'_Train.txt'
            data_train = pd.read_csv(train_file, sep='\t') #, header = 0
            timeLabel = time_label
            censorLabel = status_label
            instID = instance_label

            #Derive the attribute and phenotype array using the phenotype label
            dataFeatures_train = data_train.drop([timeLabel,censorLabel,instID],axis = 1).values
            dataEvents_train = data_train[[timeLabel,censorLabel]].values

            #Optional: Retrieve the headers for each attribute as a length n array
            dataHeaders_train = data_train.drop([timeLabel,censorLabel,instID],axis=1).columns.values

            #split dataEvents into two separate arrays (time and censoring)
            dataEventTimes_train = dataEvents_train[:,0]
            dataEventStatus_train = dataEvents_train[:,1]


            test_file = self.data_path + '/' + str(self.model_type) + '_cens'+ str(self.censor) + '_surv' + '_CV_'+str(i)+'_Test.txt'
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

            ### Open the survival_ExSTraCS model
            with open(self.model_path+'/cens_'+str(self.censor)+'/ExSTraCS_'+str(i),'rb') as file:
                self.trainedModel = pickle.load(file)
            
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



       	    # PLOT predicted times - need to also intake max time to set X axis
            self.trainedModel.plotPreds(self.predList) #plots the predicted times of the test data across all CVs
            plt.savefig(self.output_path+'/pred_hist'+self.model_type+'.svg')
            plt.close()

            # Output the individual survival distributions - fix this to call from trainedModel
            ax = self.predProbs.plot(legend = False, title = 'Individual Survival Probabilities ('+ self.model_type+' model)')
            ax.set_ylabel("Survival Probability")
            ax.set_xlabel("Time")
            plt.savefig(self.output_path+'/pred_survival_probs_'+self.model_type+'.svg')
            plt.close()

            #for KM plots of top 5 features
            self.dataFeatures = np.append(dataFeatures_train, dataFeatures_test, axis = 0)
            self.dataEvents = np.append(dataEvents_train, dataEvents_test, axis = 0)
            self.dataHeaders = dataHeaders_train

            #Plot the true event times of each model
            self.plotTrueEventTimes(self.output_path+'/' + str(self.model_type) + '_cens'+ str(self.censor) + '_surv')


            # Plot results from EACH model
            try:
                self.plot_results(self.censor)
            except Exception as ex:
                print("Plot results error: ",ex)
        
        return self.predList
    
    def plotTrueEventTimes(self, data):
        data = pd.read_csv(data, sep = '\t', header = 0)
        #print(list(data.columns))
        dataEventTimes = list(data[time_label])
        plt.figure(figsize=(6, 4))
        plt.xlim([0,max(dataEventTimes)])
        plt.xlabel('True Event Times', fontsize=14)
        plt.ylabel('# of Instances', fontsize=14)
        plt.hist(x=dataEventTimes, bins='auto', color='cadetblue', alpha=0.7, rwidth=0.85)
        plt.savefig(self.output_path+'/true_hist'+self.model_type+'.png')
        plt.close()

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
    
    def get_output(self):
        return None
