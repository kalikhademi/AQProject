# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:53:13 2018
Name: Subselect_data 
@authors: Kiana Alikhademi, Princess Lyons, Joshua Peeples, Armisha Roberts
"""

# loading libraries
from sklearn.cross_validation import train_test_split, cross_val_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import csv
import pdb
import os
from RemoveOnly import RemoveOnly

#Reads in glass identification from file 
filename = 'Glass ID_3.csv';
raw_data = open(filename, 'rt');
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
x = list(reader);
pdb.set_trace
data = np.array(x).astype('float');
print(data.shape);

#Percentange of data to select randomly
percent = .8
numTrials = 5
trial_ID = []
trial_data = []

ncolumns = len(data[1]) - 1

#Create directory to save data for each new trial
path = os.getcwd()

#Name folder based on percentage
path = path +  '_Percent_' + str(percent*100)

for i in range(numTrials):
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(data, data[:,ncolumns], test_size=(1-percent))
    
    #Save out to Excel
    #make directory 
    if not os.path.exists(path):
        os.makedirs(path)
        
    #Write training and test data to file 
    np.savetxt(path + '\\' + '\\Sub_select_Trial' + str(i) + '_Percent_' + str(percent*100) + '.csv', X_train, delimiter=",")
    #Get unique ID and add one
    Unique_ID = np.add(X_train[:,0],1)

    #Remove unique ID and only have features/labels
    X_train = np.delete(X_train,np.s_[0],axis=1)
    
    #Append Unique_ID for each trial
    trial_ID.append(Unique_ID) 
    
    #Append Trial data and labels
    trial_data.append(X_train)
