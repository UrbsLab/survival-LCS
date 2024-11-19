import time
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

iterations = 50000
random_state = 42

cv_count = 5
pmethod = "random"
isContinuous = True
nu = 1
rulepop = 1000

class ExperimentRun:

    def __init__(self, dpath, mpath, opath, mtype, cv, censor):
        self.data_path = dpath
        self.model_path = mpath
        self.output_path = opath
        self.model_type = mtype
        self.cv = cv
        self.censor = censor
        
    def run(self):
        ibs_df = pd.DataFrame()
        ibs_df['times'] = range(T + 1)
        ibs_df.set_index('times',inplace=True)
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


        start = time.time()
        ### Train the survival_ExSTraCS model
        model = survivalLCS(learning_iterations=iterations, nu=nu ,N=rulepop)
        trainedModel = model.fit(dataFeatures_train,dataEventTimes_train,dataEventStatus_train)



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

            survs = est.predict_survival_function(dataFeatures_test)
            cox_times = np.arange(max(min(dataEventTimes_test), min(dataEventTimes_train)), min(max(dataEventTimes_test),max(dataEventTimes_train)))
            preds = np.asarray([[fn(t) for t in cox_times] for fn in survs])

            try:
                times, cox_bscores = sksurv.metrics.brier_score(scoreEvents_test, scoreEvents_test, preds, cox_times)

                cb =pd.DataFrame({'times':times, 'b_scores'+str(i):cox_bscores})
                cb.set_index('times',inplace=True)

                cb_df = pd.concat([cb,cb_df],axis=1,sort=False).reset_index()
                cb_df.set_index('times',inplace=True)

            except Exception as e:
                print(e, 'No Cox brier scores generated')
        else:
            print("Comparison approaches not run on datasets with > 100 features")


        # -------------------------------------------------------------------------------------------
        ### Survival Prediction - LCS
        #--------------------------------------------------------------------------------------------
        ##HERE Make this function also generate all of the relevant graphs and save them in the appropriate output file
        if predList is None:
            predList = trainedModel.predict(dataFeatures_test)
        else:
            predList = np.append(predList, trainedModel.predict(dataFeatures_test))

        #print(predList)

        if predProbs is None:
            predProbs = pd.DataFrame(trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T
            #print(predProbs.head())
        else:
            predProbs = pd.concat([predProbs, pd.DataFrame(trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T])


        #(add i += 1?)

        #Retrieve attsums, (average across all CV or just use sum across??)
        if attSums is None:
            attSums = np.array(trainedModel.AT.getSumGlobalAttTrack(trainedModel))
            featImp = attSums #this is going to be an nd.array used to make a box plot later
        else:
            attSums = attSums + np.array(trainedModel.AT.getSumGlobalAttTrack(trainedModel))
            featImp = np.vstack((featImp, np.array(trainedModel.AT.getSumGlobalAttTrack(trainedModel))))
        #     print("featImp: ", featImp)
        #     if i == cv_count
        #     attSums / cv_count

        ### Pickle the model
        pickle.dump(trainedModel, open(model_path+'/cens_'+str(censor)+'/ExSTraCS_'+str(i),'wb'))
        #model.pickle_model(defaultExportDir+'/ExSTraCS_'+str(i))
        print("Pickled survivalLCS Model #"+str(i))

        loopend = time.time()
        runtime = loopend - start
        runtime_df.loc[len(runtime_df.index)] = [model_type,censor, output_path, dataFeatures_train.shape[1], runtime]

        #Obtain the integrated brier score
        try:
            times, b_scores = trainedModel.brier_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)

            #ibs_value = trainedModel.integrated_b_score(dataFeatures_test,dataEventStatus_test,dataEventTimes_test,dataEventTimes_train,scoreEvents_train,scoreEvents_test)
            #print("integrated_brier_score: ",ibs_value)

            tb = pd.DataFrame({'times':times, 'b_scores'+str(i):b_scores})
            tb.set_index('times',inplace=True)


            #sum, then average scores
            ibs_df = pd.concat([tb,ibs_df],axis=1,sort=False).reset_index()
            ibs_df.set_index('times',inplace=True)
        except Exception as e:
            print('Error generating integrated Brier scores', e)
            pass


        #trainedModel.plot_ibs(times, b_scores)
        #plt.savefig(output_path+'/ibs_plot_'+model_type+'cv_'+str(i)+'.svg')
        #plt.close()


    #PLOT predicted times - need to also intake max time to set X axis
    #trainedModel.plotPreds(predList) #plots the predicted times of the test data across all CVs
    #plt.savefig(output_path+'/pred_hist'+model_type+'.svg')
    #plt.close()

    #Output the individual survival distributions - fix this to call from trainedModel
    #ax = predProbs.plot(legend = False, title = 'Individual Survival Probabilities ('+ model_type+' model)')
    #ax.set_ylabel("Survival Probability")
    #ax.set_xlabel("Time")
    #plt.savefig(output_path+'/pred_survival_probs_'+model_type+'.svg')
    #plt.close()

    #for ibs plotting, generate columns mean and CI
    ibs_df[str(os.path.basename(output_path)) + '_cens'+ str(censor)] = ibs_df.mean(axis = 1)
    #print('ibs_df :', ibs_df)
    ibs_df[str(os.path.basename(output_path)) + '_cens'+ str(censor)+'_ci_lower'] = ibs_df[str(os.path.basename(output_path)) + '_cens'+ str(censor)] - (ibs_df.std(axis = 1)*2)
    ibs_df[str(os.path.basename(output_path)) + '_cens'+ str(censor)+'_ci_upper'] = ibs_df[str(os.path.basename(output_path)) + '_cens'+ str(censor)] + (ibs_df.std(axis = 1)*2)
    #ibs_df.to_csv(output_path+'/ibs_data_'+model_type+'.txt', index = False)
    print('ibs_df :', ibs_df)

    brier_df = pd.concat([brier_df,ibs_df], axis = 1, sort = False).reset_index()
    brier_df.set_index('times', inplace = True)
    brier_df = brier_df.loc[:, ~brier_df.columns.str.startswith('b_scores')]
    print('brier_df from one dataset (across 3 cvs): ', brier_df)


    try:
        cb_df[str(os.path.basename(output_path))] = cb_df.mean(axis = 1)
        cb_df[str(os.path.basename(output_path)) + '_ci_lower'] = cb_df[str(os.path.basename(output_path))] - (cb_df.std(axis = 1)*2)
        cb_df[str(os.path.basename(output_path)) + '_ci_upper'] = cb_df[str(os.path.basename(output_path))] + (cb_df.std(axis = 1)*2)

        cox_bscore_df = pd.concat([cox_bscore_df,cb_df], axis = 1, sort = False).reset_index()
        cox_bscore_df.set_index('times', inplace = True)
        cox_bscore_df = cox_bscore_df.loc[:, ~cox_bscore_df.columns.str.startswith('b_scores')]
    except Exception as e:
        print(e)
        continue



    #for KM plots of top 5 features
    dataFeatures = np.append(dataFeatures_train, dataFeatures_test, axis = 0)
    dataEvents = np.append(dataEvents_train, dataEvents_test, axis = 0)
    dataHeaders = dataHeaders_train

    #Plot the true event times of each model
    plotTrueEventTimes(output_path+'/' + str(model_type) + '_cens'+ str(censor) + '_surv')


    ## Plot results from EACH model
    try:
        plot_results(censor)
    except Exception as ex:
        print("Plot results error: ",ex)
