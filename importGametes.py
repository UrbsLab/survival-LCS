import pandas as pd
import numpy as np
import os
import ast
from itertools import chain

#Notes: This script read
"""
Name:        importGametes.py
Authors:     Alexa Woodward, written at the University of Pennsylvania, Philadelphia, PA, USA
Contact:     alexa.a.woodward@gmail.com
Created:     February 25, 2022
Modified:    ---
Description: Using gametes data and model file(s), this module automatically parses the files to return data that can be passed to the simulation script.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
Survival-LCS: XXX.  
Copyright (C) 2022 Alexa Woodward
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

'''
:param gametesData:    file name of gametes data output
:param gametesModel0:   file name of the gametes model file
:param gametesModel1:   file name of the second gametes model file (for additive and heterogeneous models)
:param model0:     Must be string. Model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"
:param model1:     Must be string. Model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"
:param full_model:     Must be string. Model type, allowed values: "main_effect, "2way_epistasis","additive", "heterogeneous"
'''

class importGametes:
    def __init__(self,gametesData,gametesModel0,model0,model1,full_model,gametesModel1=None): #could include more models eventually
        gametesData = pd.read_csv(gametesData, header = 0, sep = '\t')
        if full_model == 'heterogeneous':
            gametesData = gametesData.replace({'Model' : { 'Model_0' : 0, 'Model_1' : 1}})
        self.gametesData = gametesData.drop(columns = ['Class']) #no need for the class column
        
        if model0 or model1 or full_model == "heterogeneous":
            self.numAttributes = gametesData.shape[1]-1 #to account for model column
        else:
            self.numAttributes = gametesData.shape[1]
            
        #self.gametesData = self.gametesData.drop(columns = ['Model']) #delete this column here? no - the hetx penetrance function needs it
        all_names = list(self.gametesData.columns.values)
        self.names = [i for i in all_names if i.startswith('M')] #need to delete "model" if it's heterogeneous
        if full_model == "heterogeneous":
            self.names.remove("Model")

        self.P = None
        self.P0 = None
        self.P1 = None

#--------------------------------------------------------------------------------
# parseModelFile
#--------------------------------------------------------------------------------

    def parseModelFile(self,gametesModel0, gametesModel1,model0, model1,full_model):
        with open(gametesModel0) as f: # Default file operation mode is `r'
            if full_model != "2way_epistasis":
                lines = [15]
            else:
                lines = [15,16,17]
                
            penetrances_m0 = []    
            for i, line in enumerate(f):
                if i in lines:
                    line = list(ast.literal_eval(line))
                    penetrances_m0.append(line)
                elif i > 17: # don't read after line 17 to save time
                    break
        penetrances_m0 = list(chain.from_iterable(penetrances_m0))
        if full_model == "main_effect":
            self.P = penetrances_m0
        else:
            self.P0 = penetrances_m0
            #print('self.P0: ', self.P0)

        if gametesModel1 is not None:
            with open(gametesModel1) as g: # Default file operation mode is `r'
                if model1 == "main_effect":
                    lines = [15]
                else:
                    lines = [15,16,17]
                penetrances_m1 = []    
                for i, line in enumerate(g):
                    if i in lines:
                        line = list(ast.literal_eval(line))
                        penetrances_m1.append(line)
                    elif i > 17: # don't read after line 17 to save time
                        break   
                self.P1 = list(chain.from_iterable(penetrances_m1))
            
        return self.P, self.P0, self.P1       
                        
        #need to end up with P, P0, P1....
            
        
#--------------------------------------------------------------------------------
# formatGametes: not used
#--------------------------------------------------------------------------------
            
    def formatGametes(self): 
        return self.gametesData


#if __name__ == "__main__":
    #create the object for Class B
   # importGametes = importGametes()    
          
          
    
