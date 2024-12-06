import pandas as pd
import numpy as np
import os
import sklearn
import scipy
import random
import importGametes
import math
import itertools 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from importGametes import * 
from itertools import chain
from scipy import interpolate

# I need to be able to input the output of importGametes.py...
'''
:param gametes: object from importGametes.py
:param T:       Max time 
:param model:   Must be string. Model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"
:param knots:   Number of knots to generate baseline survival model, default = 8  
'''

class genSurvSim:
    def __init__(self,import_gametes,T,model,distribution_type="randomspline"): # I need this to parse the gametes model and dataset files
        
        self.model = model
        self.X = import_gametes.gametesData
        self.P = import_gametes.P
        self.P0 = import_gametes.P0
        self.P1 = import_gametes.P1
        self.survival = None
        self.baseline = None
        self.T = T
        self.d_type = distribution_type

    def baseline_build(self,T,distribution_type="randomspline"):
        if distribution_type.lower() == "gaussian":
            return self.baseline_build_normal(self.T)
        elif distribution_type.lower() == "gamma":
            return self.baseline_build_gamma(self.T)
        elif distribution_type.lower() == "weibull":
            return self.baseline_build_weibull(self.T)
        elif distribution_type.lower() == "random": 
            return self.baseline_build_random(self.T)
        elif distribution_type.lower() == "randomspline": 
            return self.baseline_build_random_spline(self.T)
        else:
            raise ValueError("Invalid distribution type. Choose from: 'gaussian', 'lognormal', 'weibull', 'random_spline'.")

    def baseline_build_random_spline(self,T):
        knots = 8
        time = range(1,(T+1))
        k = sorted([1, T+1] +  random.sample(range(2, T), knots)) #sample a range of values (= to #of knots) starting with 1 and ending with T+1
        heights = sorted([0,1] + list(np.random.uniform(size = knots))) #sample from the uniform distribution, bookend by o and 1
        tk = pd.DataFrame(
            {'time' : k,
             'heights' : heights
            })
        MonotonicSpline = scipy.interpolate.PchipInterpolator(tk.iloc[:,0], tk.iloc[:,1], extrapolate=True)
        bl_failure_CDF = MonotonicSpline(time).tolist() + [1]
        baseline = pd.DataFrame(
            {'time': range(1,T+1),
             'failure_PDF': np.diff(bl_failure_CDF),
             'failure_CDF': bl_failure_CDF[1:],
             'survivor': abs(1 - np.array(bl_failure_CDF[1:]))
            })
        baseline['hazard'] = baseline['failure_PDF']/(1 - np.array(bl_failure_CDF[:T]))
        
        self.baseline = baseline
        return self.baseline
    
    def baseline_build_random(self, T):
        """
        Build baseline survival data using a random distribution.

        :param T: Max time
        :return: Baseline survival data as a DataFrame
        """
        time = np.arange(1, T + 1)
        
        # Generate random failure probabilities
        failure_PDF = np.random.random(size=T)
        
        # Normalize the failure PDF to sum to 1 for proper probabilities
        failure_PDF = failure_PDF / np.sum(failure_PDF)
        
        # Calculate the CDF from the PDF
        failure_CDF = np.cumsum(failure_PDF)
        
        # Survival function is 1 - CDF
        survivor = 1 - failure_CDF
        
        # Hazard function is PDF / survival
        hazard = failure_PDF / survivor
        hazard[survivor == 0] = 0  # Avoid division by zero
        
        # Create the baseline DataFrame
        baseline = pd.DataFrame({
            'time': time,
            'failure_PDF': failure_PDF,
            'failure_CDF': failure_CDF,
            'survivor': survivor,
            'hazard': hazard
        })
        
        self.baseline = baseline
        return self.baseline

    def baseline_build_normal(self, T, mu=0.5, sigma=0.1):
        """
        Build baseline survival data using a normal distribution.

        :param T: Max time
        :param knots: Number of knots (not used in this implementation, but kept for compatibility)
        :param mu: Mean of the normal distribution
        :param sigma: Standard deviation of the normal distribution
        :return: Baseline survival data as a DataFrame
        """
        time = np.arange(1, T + 1)
        
        # Generate failure probabilities using the normal distribution's PDF
        failure_PDF = scipy.stats.norm.pdf(time, loc=mu * T, scale=sigma * T)
        
        # Normalize the failure PDF to sum to 1 for proper probabilities
        failure_PDF = failure_PDF / np.sum(failure_PDF)
        
        # Calculate the CDF from the PDF
        failure_CDF = np.cumsum(failure_PDF)
        
        # Survival function is 1 - CDF
        survivor = 1 - failure_CDF
        
        # Hazard function is PDF / survival
        hazard = failure_PDF / survivor
        hazard[survivor == 0] = 0  # Avoid division by zero
        
        # Create the baseline DataFrame
        baseline = pd.DataFrame({
            'time': time,
            'failure_PDF': failure_PDF,
            'failure_CDF': failure_CDF,
            'survivor': survivor,
            'hazard': hazard
        })
        
        self.baseline = baseline
        return self.baseline

    def baseline_build_weibull(self, T, shape=0.63, scale=0.33):
        """
        Build baseline survival data using a Weibull distribution.

        :param T: Max time
        :param shape: Shape parameter (k) of the Weibull distribution
        :param scale: Scale parameter (λ) of the Weibull distribution
        :return: Baseline survival data as a DataFrame
        """
        time = np.arange(1, T + 1)
        
        # Generate failure probabilities using the Weibull distribution's PDF
        failure_PDF = scipy.stats.weibull_min.pdf(time, c=shape, scale=scale * T)
        
        # Normalize the failure PDF to sum to 1 for proper probabilities
        failure_PDF = failure_PDF / np.sum(failure_PDF)
        
        # Calculate the CDF from the PDF
        failure_CDF = np.cumsum(failure_PDF)
        
        # Survival function is 1 - CDF
        survivor = 1 - failure_CDF
        
        # Hazard function is PDF / survival
        hazard = failure_PDF / survivor
        hazard[survivor == 0] = 0  # Avoid division by zero
        
        # Create the baseline DataFrame
        baseline = pd.DataFrame({
            'time': time,
            'failure_PDF': failure_PDF,
            'failure_CDF': failure_CDF,
            'survivor': survivor,
            'hazard': hazard
        })
        
        self.baseline = baseline
        return self.baseline
    
    def baseline_build_gamma(self, T, shape=2.0, scale=10.0):
        """
        Build baseline survival data using a Gamma distribution.

        :param T: Max time
        :param shape: Shape parameter (k) of the Gamma distribution
        :param scale: Scale parameter (θ) of the Gamma distribution
        :return: Baseline survival data as a DataFrame
        """
        # Generate time points
        time = np.arange(1, T + 1)
        
        # Calculate the failure CDF using the Gamma distribution
        failure_CDF = scipy.stats.gamma.cdf(time, a=shape, scale=scale)
        
        # Ensure CDF is normalized (should already be, but to be consistent)
        failure_CDF = np.clip(failure_CDF, 0, 1)
        
        # Calculate the failure PDF
        failure_PDF = np.diff(np.concatenate(([0], failure_CDF)))
        
        # Calculate the survival function
        survivor = 1 - failure_CDF
        
        # Calculate the hazard function
        hazard = failure_PDF / survivor
        hazard[survivor == 0] = 0  # Avoid division by zero
        
        # Create the baseline DataFrame
        baseline = pd.DataFrame({
            'time': time,
            'failure_PDF': failure_PDF,
            'failure_CDF': failure_CDF,
            'survivor': survivor,
            'hazard': hazard
        })
        
        self.baseline = baseline
        return self.baseline

    def penetrance_mainEffect(self,X, P, names, sd = 0.1):
        if (X[names[0]] == 0):
            p = np.random.normal(P[0],sd)
        elif (X[names[0]] == 1):
            p = np.random.normal(P[1],sd)
        elif (X[names[0]] == 2):
            p = np.random.normal(P[2],sd)     
        else:
            p = 0
        if 0 < 0: 
            p = min(P)
        elif p > 1:
            p = max(P)
                
        return p

    def penetrance_2way(self, X, P, names, sd = 0.05):
        if ((X[names[0]] == 0) & (X[names[1]] == 0)):
            p = np.random.normal(P[0],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 0)):
            p = np.random.normal(P[1],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 0)):
            p = np.random.normal(P[2],sd)  
        elif ((X[names[0]] == 0) & (X[names[1]] == 1)):
            p = np.random.normal(P[3],sd)    
        elif ((X[names[0]] == 1) & (X[names[1]] == 1)):
            p = np.random.normal(P[4],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 1)):
            p = np.random.normal(P[5],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 2)):
            p = np.random.normal(P[6],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 2)):
            p = np.random.normal(P[7],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 2)):
            p = np.random.normal(P[8],sd)
        else:
            p = 0
            
        if 0 < 0: 
            p = min(P)
        elif p > 1:
            p = max(P)
            
        return p

    def penetrance_add(self,X, P0, P1, names, sd = 0.05):
        P01 = []
        for i in P0:
            for j in P1:
                P01.append((i+j)/2) #attempting average
        
        P = []
        for i in P01: 
            P.append(i)
        
        if ((X[names[0]] == 0) & (X[names[1]] == 0) & (X[names[2]] == 0)):
                p = np.random.normal(P[0],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 0)& (X[names[2]] == 1)):
            p = np.random.normal(P[1],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 0)& (X[names[2]] == 2)):
            p = np.random.normal(P[2],sd) 
        elif ((X[names[0]] == 0) & (X[names[1]] == 1)& (X[names[2]] == 0)):
            p = np.random.normal(P[3],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 1)& (X[names[2]] == 1)):
            p = np.random.normal(P[4],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 1)& (X[names[2]] == 2)):
            p = np.random.normal(P[5],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 2)& (X[names[2]] == 0)):
            p = np.random.normal(P[6],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 2)& (X[names[2]] == 1)):
            p = np.random.normal(P[7],sd)
        elif ((X[names[0]] == 0) & (X[names[1]] == 2)& (X[names[2]] == 2)):
            p = np.random.normal(P[8],sd)   
        elif ((X[names[0]] == 1) & (X[names[1]] == 0) & (X[names[2]] == 0)):
            p =np.random.normal(P[9],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 0)& (X[names[2]] == 1)):
            p = np.random.normal(P[10],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 0)& (X[names[2]] == 2)):
            p = np.random.normal(P[11],sd) 
        elif ((X[names[0]] == 1) & (X[names[1]] == 1)& (X[names[2]] == 0)):
            p = np.random.normal(P[12],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 1)& (X[names[2]] == 1)):
            p = np.random.normal(P[13],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 1)& (X[names[2]] == 2)):
            p = np.random.normal(P[14],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 2)& (X[names[2]] == 0)):
            p = np.random.normal(P[15],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 2)& (X[names[2]] == 1)):
            p = np.random.normal(P[16],sd)
        elif ((X[names[0]] == 1) & (X[names[1]] == 2)& (X[names[2]] == 2)):
            p = np.random.normal(P[17],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 0) & (X[names[2]] == 0)):
            p = np.random.normal(P[18],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 0)& (X[names[2]] == 1)):
            p = np.random.normal(P[19],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 0)& (X[names[2]] == 2)):
            p = np.random.normal(P[20],sd) 
        elif ((X[names[0]] == 2) & (X[names[1]] == 1)& (X[names[2]] == 0)):
            p = np.random.normal(P[21],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 1)& (X[names[2]] == 1)):
            p = np.random.normal(P[22],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 1)& (X[names[2]] == 2)):
            p = np.random.normal(P[23],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 2)& (X[names[2]] == 0)):
            p = np.random.normal(P[24],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 2)& (X[names[2]] == 1)):
            p = np.random.normal(P[25],sd)
        elif ((X[names[0]] == 2) & (X[names[1]] == 2)& (X[names[2]] == 2)):
            p = np.random.normal(P[26],sd)       
        else:
            p = 0
        return p

    def penetrance_het(self, X, P0, P1, names, sd = 0.05):
        if (X["Model"] == 0): #need to fix this 
            if (X[names[0]] == 0):
                p = np.random.normal(P0[0],sd)
            elif (X[names[0]] == 1):
                p = np.random.normal(P0[1],sd)
            elif (X[names[0]] == 2):
                p = np.random.normal(P0[2],sd)       
            else:
                p = 0
        else:
            if ((X[names[1]] == 0) & (X[names[2]] == 0)):
                p = np.random.normal(P1[0],sd)
            elif ((X[names[1]] == 0)& (X[names[2]] == 1)):
                p = np.random.normal(P1[1],sd)
            elif ((X[names[1]] == 0)& (X[names[2]] == 2)):
                p = np.random.normal(P1[2],sd) 
            elif ((X[names[1]] == 1)& (X[names[2]] == 0)):
                p = np.random.normal(P1[3],sd)
            elif ((X[names[1]] == 1)& (X[names[2]] == 1)):
                p = np.random.normal(P1[4],sd)
            elif ((X[names[1]] == 1)& (X[names[2]] == 2)):
                p = np.random.normal(P1[5],sd)
            elif ((X[names[1]] == 2)& (X[names[2]] == 0)):
                p = np.random.normal(P1[6],sd)
            elif ((X[names[1]] == 2)& (X[names[2]] == 1)):
                p = np.random.normal(P1[7],sd)
            elif ((X[names[1]] == 2)& (X[names[2]] == 2)):
                p = np.random.normal(P1[8],sd)        
            else:
                p = 0 
        return p

    def generate_time(self, X, names, P,P0,P1, model,censor = 0.1): #adding "model" to specify which penetrance function to use

        baseline = self.baseline_build(self.T, self.d_type)
        
        T = max(baseline['time'])
        X = pd.DataFrame(X)
        if model == "main_effect":
            X['p'] = X.apply(lambda x: self.penetrance_mainEffect(x, P, names, sd = 0.1), axis=1)
        elif model == "2way_epistasis":
            X['p'] = X.apply(lambda x: self.penetrance_2way(x, P0, names, sd = 0.1), axis=1)
        elif model == "additive":
            X['p'] = X.apply(lambda x: self.penetrance_add(x, P0, P1, names, sd = 0.1), axis=1)
        elif model == "heterogeneous":
            X['p'] = X.apply(lambda x: self.penetrance_het(x, P0, P1, names, sd = 0.1), axis=1)
        else:
            print("Error: model type not recognized")
        #this normalizes the data back between zero and 1 in case any of the values are outside of those bounds     
        X['p'] = (X['p'] - X['p'].min()) / (X['p'].max() - X['p'].min())
        XP = pd.DataFrame(X['p'])
        survival = XP.apply(lambda x: baseline['survivor']**math.exp(x), axis = 1)
        cols = list(survival.columns)
        #survival.insert(0, 'V1', 1)
        
        y = []
        i = 0
        for pen in X['p']:
            y.append(np.argmax(survival.iloc[i] < pen))
            i +=1 
        
        for j in range(len(y)):
            if y[j] == 0:
                y[j] = 1
                
        #y = survival.apply(lambda x: np.argmax(x < x.p) + 1, axis = 1)
        X = X.drop(columns = ['p'])             
        X = pd.DataFrame(X)
        X['eventTime'] = y
        X['eventStatus'] = X.apply(lambda x: np.random.uniform() > censor, axis = 1).astype(int)
        #drop the model column if a heterogeneous dataset 
        if model == "heterogeneous":
            X = X.drop(columns = ['Model'])
        #add instance label column for later! Need for LCS_DIVE    
        X["inst"] = X.index + 1    

        self.survival = pd.DataFrame(survival).T
        
        return X, self.survival

    def plot_survival_dists(self):
        ax = self.survival.plot(legend = False, title = 'Individual Survival Probabilities')
        ax.set_ylabel("Survival Probability")
        ax.set_xlabel("Time")
