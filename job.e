slurmstepd: error: execve(): run_full_pipeline.sh: Permission denied
srun: error: esplhpc-cp019: task 0: Exited with exit code 13
Traceback (most recent call last):
  File "/common/bandheyh/survival-lcs/sim_full_pipeline_sLCS_final.py", line 175, in <module>
    survivalLCS.returnCVModelFiles()
  File "/common/bandheyh/survival-lcs/survival_LCS_pipeline.py", line 417, in returnCVModelFiles
    self.predProbs = pd.concat([self.predProbs, pd.DataFrame(self.trainedModel.predict_proba(dataFeatures_test, dataEventTimes_test)).T])
  File "/common/bandheyh/survival-lcs/survival_ExSTraCS.py", line 672, in predict_proba
    return np.array(predList)
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (200,) + inhomogeneous part.
srun: error: esplhpc-cp019: task 0: Exited with exit code 1
