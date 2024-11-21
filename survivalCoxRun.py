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
knots = 8

iterations = 200000
random_state = 42

cv_count = 5
pmethod = "random"
isContinuous = True
nu = 1
rulepop = 2000

class ExperimentRun:

    def __init__(self, dpath, mpath, opath, mtype, cv, censor):
        self.data_path = dpath
        self.model_path = mpath
        self.output_path = opath
        self.model_type = mtype
        self.cv = cv
        self.censor = censor
        
    def run(self):
        cb_df = pd.DataFrame()
        cb_df['times'] = range(T + 1)
        cb_df.set_index('times',inplace=True)

        model_path_censor = self.model_path + '/cens_'+ str(self.censor)
        make_folder(model_path_censor)

        output_path_censor = self.output_path + '/cens_'+ str(self.censor)
        make_folder(output_path_censor)

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

        cb = None

        # Cox Proportional Hazards Model - maybe run this only if features <= 100
        if dataFeatures_train.shape[1] < 101:
            CoxPH = make_pipeline(CoxPHSurvivalAnalysis(alpha = 0.00001))
            est =  CoxPH.fit(dataFeatures_train, scoreEvents_train)

            survs = est.predict_survival_function(dataFeatures_test)
            cox_times = np.arange(max(min(dataEventTimes_test), min(dataEventTimes_train)), min(max(dataEventTimes_test),max(dataEventTimes_train)))
            preds = np.asarray([[fn(t) for t in cox_times] for fn in survs])

            try:
                times, cox_bscores = sksurv.metrics.brier_score(scoreEvents_test, scoreEvents_test, preds, cox_times)

                col_name = 'b_scores_' + \
                    str(os.path.basename(self.output_path)) + \
                    '_cens'+ str(self.censor) + \
                            '_cv' + str(self.cv)
                
                cb = pd.DataFrame({'times':times, col_name:cox_bscores})
                
                temp_df = cb.copy()
                temp_df = temp_df.dropna()

                cb.set_index('times',inplace=True)

                cb.to_csv(self.output_path+'/cens_'+str(self.censor)+'/CoxModel_'+str(self.cv)+'_brierscores.csv')

                try:
                    ibs_value = np.trapz(temp_df[col_name], temp_df['times']) / (list(temp_df['times'])[-1] - list(temp_df['times'])[0])
                except Exception as e:
                    ibs_value = np.nan
                    raise e
                
                print("integrated_brier_score: ", ibs_value)
                with open(self.output_path+'/cens_'+str(self.censor)+'/CoxModel_'+str(self.cv)+'_ibsscore.txt','w') as file:
                    file.write("Model, " + col_name + '\n')
                    file.write("Integrated Brier Score, " + str(ibs_value))

            except Exception as e:
                print(e, 'No Cox brier scores generated')
                raise e
        else:
            print("Comparison approaches not run on datasets with > 100 features")

        return cb
