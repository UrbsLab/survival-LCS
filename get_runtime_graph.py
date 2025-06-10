# %%
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
model, nfeat, maf, censoring, cv = 'het', 'f10000', 'maf0.4', '0.1', '0'

# %%
objects = []
with (open('pipeline/pickled_cv_models/'+model+'/'+model+'_'+nfeat+'_'+maf+'/cens_'+censoring+'/ExSTraCS_'+cv, "rb")) as file:
    while True:
        try:
            objects.append(pickle.load(file))
        except EOFError:
            break

# %%
trainedModel = objects[0]
trainedModel.export_iteration_tracking_data("output/iterationData.csv")

def cumulativeFreq(freq):
    a = []
    c = []
    for i in freq:
        a.append(i+sum(c))
        c.append(i)
    return np.array(a)

def movingAvg(a,threshold=300):
    weights = np.repeat(1.0,threshold)/threshold
    conv = np.convolve(a,weights,'valid')
    return np.append(conv,np.full(threshold-1,conv[conv.size-1]),)

dataTracking = pd.read_csv("output/iterationData.csv")

iterations = dataTracking["Iteration"].values
gTime = dataTracking["Total Global Time"].values
mTime = dataTracking["Total Matching Time"].values
covTime = dataTracking["Total Covering Time"].values
crossTime = dataTracking["Total Crossover Time"].values
covTime = dataTracking["Total Covering Time"].values
mutTime = dataTracking["Total Mutation Time"].values
atTime = dataTracking["Total Attribute Tracking Time"].values
initTime = dataTracking["Total Model Initialization Time"].values
rcTime = dataTracking["Total Rule Compaction Time"].values
delTime = dataTracking["Total Deletion Time"].values
subTime = dataTracking["Total Subsumption Time"].values
selTime = dataTracking["Total Selection Time"].values
evalTime = dataTracking["Total Evaluation Time"].values

color_list = [None,'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','m','k']
plt.plot(iterations,initTime,label="Init Time", color=color_list[1])
plt.plot(iterations,mTime+initTime,label="Matching Time",color=color_list[2])
plt.plot(iterations,covTime+mTime+initTime,label="Covering Time",color=color_list[3])
plt.plot(iterations,selTime+covTime+mTime+initTime,label="Selection Time",color=color_list[4])
plt.plot(iterations,crossTime+selTime+covTime+mTime+initTime,label="Crossover Time",color=color_list[5])
plt.plot(iterations,mutTime+crossTime+selTime+covTime+mTime+initTime,label="Mutation Time",color=color_list[6])
plt.plot(iterations,subTime+mutTime+crossTime+selTime+covTime+mTime+initTime,label="Subsumption Time",color=color_list[7])
plt.plot(iterations,atTime+subTime+mutTime+crossTime+selTime+covTime+mTime+initTime,label="AT Time",color=color_list[8])
plt.plot(iterations,delTime+atTime+subTime+mutTime+crossTime+selTime+covTime+mTime+initTime,label="Deletion Time",color=color_list[9])
plt.plot(iterations,rcTime+delTime+atTime+subTime+mutTime+crossTime+selTime+covTime+mTime+initTime,label="RC Time",color=color_list[10])
plt.plot(iterations,evalTime+rcTime+delTime+atTime+subTime+mutTime+crossTime+selTime+covTime+mTime+initTime,label="Evaluation Time",linestyle='--',color=color_list[11])
plt.plot(iterations,gTime,label="Total Time",linestyle='--',color=color_list[12])
plt.xlabel('Iteration')
plt.ylabel('Cumulative Time (Seconds)')
handles, labels = plt.gca().get_legend_handles_labels()
max_values = [np.max(line.get_ydata()) for line in handles]
sorted_indices = np.argsort(max_values)[::-1]
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
plt.legend(sorted_handles, sorted_labels)
plt.savefig('../output/Figure4.png', dpi=600)
plt.show()

# %%
os.remove("output/iterationData.csv")


