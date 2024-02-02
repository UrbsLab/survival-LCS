"""
Name:        cv_partitioner.py
Authors:     Ryan Urbanowicz - Written at the University of Pennsylvania, Philadelphia, PA
Contact:     ryanurb@upenn.edu
Created:     10/6/18
Description: This script creates k-fold datasets for cross validation (CV), and saves them as separate .txt files.
             This script takes in a single dataset with a header, feature columns, and an endpoint column.
Required:    At minimum '-i' parameter must be specified (i.e. dataset name and/or path)
Example:    python ./cv_partitioner.py -h
            python ./cv_partitioner.py -i 6Multiplexer_Data_Complete.txt
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
import sys
import os
import argparse
from random import shuffle
from random import sample

class cvPartitioner():
    def __init__(self, data_path, output_path, folds, pmethod,outcomeLabel = 'eventTime',sortbins = 3,matchingLabel = ''):
        #Inputs: (1) dataset path/name, (2) destination folder, (3) k-folds, and (4) CV method
        #Argument Parsing :------------------------------------------------------------------------------------
        #parser = argparse.ArgumentParser(description='Create CV partitioned datasets and/or datasets with a permuted endpoint.')
        #parser.add_argument('-i', '--datapath', help='Str: File path/name for target dataset', type=str)
        #parser.add_argument('-o', '--outputpath', help='Str: Folder path/name for target dataset', type=str, default='')
        #parser.add_argument('-k', '--folds', help='Int: number of CV partitions/folds', type=int, default=10)
        #parser.add_argument('-p', '--pmethod', help='Str: Specify the partitioning approach from the following options (random, stratified, matched)', type=str, default='stratified')
        #parser.add_argument('-l', '--outcomelabel', help='Str: Label identifying the outcome/class column', type=str, default='Class')
        #parser.add_argument('-c','--isContinuous', help='Boolean: Specify if the endpoint is continuous-valued.', action='store_true')
        #parser.add_argument('-b', '--sortbins', help='Int: number of bins used for stratified partitioning with continuous endpoint data', type=int, default=3)
        #parser.add_argument('-m', '--matchinglabel', help='Str: Label for a variable in dataset that identifies matched rows that should be kept together withing CV partitions', type=str, default='')



        self.data_path = data_path
        self.output_path = output_path
        self.folds = folds
        self.pmethod = pmethod
        self.matchingLabel = matchingLabel
        self.outcomeLabel = outcomeLabel
        self.datasetList = []
        self.isContinuous = True
        self.sortbins = sortbins
        
        # Parse the dataset name
        data_name = ''
        dataList = self.data_path
        if len(dataList) > 1: #path given
            data_name = dataList.split('/')[-1]
            print(data_name)
            #data_name = dataList[-1]
        else:
            data_name = dataList[0]
            
        data_name = data_name.strip('.txt')
        
        #Load data
        headerList, datasetList = self.loadData(self.data_path)
        print('Dataset Dimensions:')
        print("Rows = "+str(len(datasetList)))
        print("Columns = "+str(len(datasetList[0])))
        
        #Get endpoint column index
        endpointIndex = headerList.index(self.outcomeLabel)
        
        #Perform initial shuffle of dataset to remove any initial bias. 
        shuffle(datasetList)  #shuffles rows
        #Run selected partitioning method
        partList = None
        
        if self.pmethod == 'random':
            partList = self.randomPartitioner()
            
        elif self.pmethod == 'stratified':
            partList = self.stratifiedPartitioner(endpointIndex, self.sortbins)
            
        elif self.pmethod == 'matched':
            partList = self.matchedPartitioner(endpointIndex, headerList)
        
        else:
            print("Error: Specified partitioning method is not available.")
        
        #Create output folder if doesn't exist, Automatically create a folder labeled (CV) 
        self.makeCVDatasets(partList,headerList,data_name,self.pmethod)
        
        
    def loadData(self,datapath):
        """ Opens a data file and saves it to a nested list. """
        f = open(datapath, 'r')
        #headerList = f.readline().rstrip('\n')  #strip off first row
        headerList = f.readline().strip('\n').split('\t')  #strip off first row
        for line in f:
            lineList = line.strip('\n').split('\t')
            self.datasetList.append(lineList)
        f.close()
        
        return headerList, self.datasetList; #return tuple including header and data list. 

    def getClasses(self, endpointIndex):
        """ Creates a list of the unique class labels in a given dataset.  Only for discrete classes."""
        classList = []
        for each in self.datasetList:
            if each[endpointIndex] not in classList:
                classList.append(each[endpointIndex])
        return classList

    def getMatches(self, matchIndex):
        """ Creates a list of the unique class labels in a given dataset.  Only for discrete classes."""
        matchList = []
        for each in self.datasetList:
            if each[matchIndex] not in matchList:
                matchList.append(each[matchIndex])
        return matchList
        
    def randomPartitioner(self):
        """ For discrete endpoints, puts an equal or near equal number of each classes within each partition. 
        For continuous endpoints, bins values into high, mid, and low, and puts an equal or near equal 
        number number from each bin in each partition. """
        #Initialize partitions
        partList = [] #stores all partitions
        for x in range(self.folds):
            partList.append([])
            
        currPart = 0
        counter = 0
        for row in self.datasetList:
            partList[currPart].append(row)
            counter += 1
            currPart = counter%self.folds
            
        return partList

    def stratifiedPartitioner(self, endpointIndex, sortbins):
        """ For discrete endpoints, puts an equal or near equal number of each classes within each partition. 
        For continuous endpoints, bins values into high, mid, and low, and puts an equal or near equal 
        number number from each bin in each partition. """
        #Initialize partitions
        partList = [] #stores all partitions
        for x in range(self.folds):
            partList.append([])
        
        if self.isContinuous: # Do stratified partitioning for continuous endpoint data
            print("Continuous Valued Endpoint Stratification")
            #Sort dataset by the target continuous endpoint column
            sortedDatasetList = sorted(self.datasetList, key=lambda x: x[endpointIndex])
            #Bin rows based on 'similar endpoint value
            byBinRows = [ [] for i in range(sortbins) ] #initialize bins
            binRemainder = len(sortedDatasetList)%sortbins
            minBinSize = int((len(sortedDatasetList) - binRemainder)/sortbins)#Get minimum bin size
            binSizeList = [ minBinSize for i in range(sortbins) ]
            
            if binRemainder > 0:
                #pick random bins to have a slightly larger size in there are not a cleanly divisible number
                targets = sample(set([i for i in range(0,sortbins)]), binRemainder)
                for each in targets:
                    binSizeList[each] += 1
            
            #fill bins with rows. 
            counter = 0
            for i in range(0,sortbins):
                start = counter
                end = counter+binSizeList[i]
                for j in range(start,end):
                    byBinRows[i].append(sortedDatasetList[counter])
                counter +=1
            
            for binSet in byBinRows:
                currPart = 0
                counter = 0
                for row in binSet:
                    partList[currPart].append(row)
                    counter += 1
                    currPart = counter%folds
                    
            return partList
            
        else: # Do stratified partitioning for binary or multiclass data
            print("Discrete Valued Endpoint Stratification")
            #Create data sublists, each having all rows with the same class
            classList = getClasses(datasetList, endpointIndex)
            byClassRows = [ [] for i in range(len(classList)) ] #create list of empty lists (one for each class)
            for row in datasetList:
                #find index in classList corresponding to the class of the current row. 
                cIndex = classList.index(row[endpointIndex])
                byClassRows[cIndex].append(row)
            
            for classSet in byClassRows:
                currPart = 0
                counter = 0
                for row in classSet:
                    partList[currPart].append(row)
                    counter += 1
                    currPart = counter%folds
                    
            return partList
          
          
    def matchedPartitioner(self, endpointIndex, headerList):
        """ Only designed for discrete endpoints. This approach relies on there being an extra 'matching' identifier
            That identifies a set of instances/rows that were matched for similar covariate values, but that have different
            class values. This partitioning approach ensures that these match sets are preserved within any given partition
            so that the matched dataset assembly is not disrupted by the cross validation proceedure. Note that this 
            implementation does not actively try to ensure that partitions are as equally sized as possible, in the case where
            not all match sets are the same size. This should not be a major issue, but one that the user should be aware of.
            This approach also assumes that each match set includes instances representative of each class, and that their 
            count ratio is roughly uniform across all match sets. """
            
        if self.isContinuous:
            print("Error: Matched partitioning only designed for discrete endpoints. ")
            pass
        
        else:
            #Get match variable column index
            matchIndex = headerList.index(self.matchinglabel)
            #Initialize partitions
            partList = [] #stores all partitions
            for x in range(self.folds):
                partList.append([])
                
            print("Discrete Valued Endpoint Stratification for matched rows.")
            #Create data sublists, each having all rows with the same match identifier
            matchList = getMatches(self.datasetList, matchIndex)
            byMatchRows = [ [] for i in range(len(matchList)) ] #create list of empty lists (one for each match group)
            for row in datasetList:
                #find index in matchList corresponding to the matchset of the current row. 
                mIndex = matchList.index(row[matchIndex])
                byMatchRows[mIndex].append(row)
            
            currPart = 0
            counter = 0
            for matchSet in byMatchRows: #Go through each unique set of matched instances
                for row in matchSet: #put all of the instances
                    partList[currPart].append(row)
                #move on to next matchset being placed in the next partition. 
                counter += 1
                currPart = counter%folds
                
        return partList
        
    def makeCVDatasets(self,partList,headerList,data_name,pmethod):  
        """ Takes in partitions of the data and creates k-folds of training and testing datasets saved as .txt files. """
        mname = None
        if pmethod == 'random':
            mname = 'R'
        elif pmethod == 'stratified':
            mname = 'S'
        elif pmethod == 'matched':
            mname = 'M'
        else:
            print("Error: Specified partitioning method is not available.")
        
        if self.output_path != '':
            #If the target folder doesn't exist yet, create it in the given path.
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            
            filePath = self.output_path+'/'+data_name

            
        for part in range(0, self.folds): #Builds CV data files.

            if not os.path.exists(filePath+'_CV_'+str(part)+'_Train.txt') or not os.path.exists(filePath+'_CV_'+str(part)+'_Test.txt'):
                print("Making new CV files:  "+filePath+'_CV_'+str(part))
                trainFile = open(filePath+'_CV_'+str(part)+'_Train.txt','w')
                testFile = open(filePath+'_CV_'+str(part)+'_Test.txt','w')
                testFile.write("\t".join(headerList)+ "\n")
                trainFile.write("\t".join(headerList)+ "\n")

                testList=partList[part] # Assign testing set as the current partition
                
                trainList=[]
                tempList = []                 
                for x in range(0,self.folds): 
                    tempList.append(x)                            
                tempList.pop(part)

                for v in tempList: #for each training partition
                    trainList.extend(partList[v])    
            
                for i in testList: #Write to Test Datafile
                    tempString = ''
                    for point in range(len(i)):
                        if point < len(i)-1:
                            tempString = tempString + str(i[point])+"\t"
                        else:
                            tempString = tempString +str(i[point])+"\n"                        
                    testFile.write(tempString)
                          
                for i in trainList: #Write to Train Datafile
                    tempString = ''
                    for point in range(len(i)):
                        if point < len(i)-1:
                            tempString = tempString + str(i[point])+"\t"
                        else:
                            tempString = tempString +str(i[point])+"\n"                        
                    trainFile.write(tempString)
                                                    
                trainFile.close()
                testFile.close()  

