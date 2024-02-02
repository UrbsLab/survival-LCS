import pandas as pd
import numpy as np

"""
Name:        calcHeritability.py
Authors:     Alexa Woodward, written at the University of Pennsylvania, Philadelphia, PA, USA
Contact:     alexa.a.woodward@gmail.com
Created:     February 25, 2022
Modified:    ---
Description: Using gametes data file(s), penetrance values, and population prevalence, this module calulates the heritability of the model.
             
-----
"""

class calcHeritability:
    def __init__(self,penetrance, K, MAFs, gametesData):

        self.sumMLG = None
        self.h2 = None
        self.K = K
        self.penetrance = penetrance
        self.MAFs = MAFs #a list, minor allele freq for each relevant SNP
        self.sumModelMLG = None
        self.X = gametesData
        
        #sum of the penetrance * probability of each MLG
        #Get relevant combinations of column names 
        all_names = list(gametesData.columns.values)
        self.names = [i for i in all_names if i.startswith('M')] #return the MLG

        
#-------------------------------------------------------------------------------------------
# modelMLG: calculates the sum of the matrix of genotype probabilities for the specified model
#-------------------------------------------------------------------------------------------
    def modelMLG(self): #pd.DataFrame(np.outer(L2, L1))

        penK = []
        for i in range(len(self.penetrance)):
            penK.append(pow(self.penetrance[i] - self.K, 2))
        if len(self.names) > 1:
            pk_matrix = pd.DataFrame(np.reshape(penK, (3,3))) #change this to work with more than just 2 SNPs?
        else:
            pk_matrix = pd.DataFrame(penK)
            

        lists = [[] for snps in range(len(self.names))]

        i = 0
        for each in self.names:
            q = self.MAFs[i]
            p = 1 - q
            p2 = pow(p,2)
            q2 = pow(q,2)
            twopq = 2*p*q
            lists[i] = [p2,twopq,q2]
            i += 1    

        if len(self.names) > 1:
            matrix = pd.DataFrame(np.outer(*lists))
        else:
            matrix = pd.DataFrame(lists[0])

        self.sumModelMLG = matrix.mul(pk_matrix).values.sum()         
        
        

#-------------------------------------------------------------------------------------------
# sum_MLG: use the gametes data directly and calculate P(G) for each, sum the values of P(G)(fG - K)^2 for each MLG
#-------------------------------------------------------------------------------------------
    def sum_MLG(self):
        if len(self.names) > 1:
            df = self.X.groupby(names).size().reset_index().rename(columns={0:'count'})
            df.reset_index(inplace=True)   
            df = df.rename(columns = {'index':'MLG'})
            df['P(G)'] = df['count'] / sum(df['count'])
        else: #if a main effect only, count the number of unique values (0/1/2)
            df = pd.DataFrame(self.X[self.names[0]].value_counts(normalize=True)).sort_index()
            #df.reset_index(inplace=True)
            df = df.rename(columns = {self.names[0]:'P(G)'})
        #sum the values P(G)(fG - K)^2 for each MLG
        sum_pen = 0
        for i in range(len(self.penetrance)):
            sum_pen += df['P(G)'][i] * pow((self.penetrance[i] - self.K), 2)

        self.sumMLG = sum_pen
        

#-------------------------------------------------------------------------------------------
# calcH: calculate the heritability h2 using the penetrance and sum of the MLGs
#-------------------------------------------------------------------------------------------
    def calcH(self):
        
        self.modelMLG()
        self.sum_MLG()
        h2_model = self.sumModelMLG / (self.K * (1 - self.K))
        h2_data = self.sumMLG / (self.K * (1 - self.K))

        return print("Model Heritability: ",h2_model,'\n',"Data Heritability: ",h2_data)   
