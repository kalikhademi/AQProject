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

def SubSelect(percentage,numTrials,data):

    #Percentange of data to select randomly
    percent = percentage;
    trial_data = [];
    
    ncolumns = len(data[1]) - 1;
    
    #Create directory to save data for each new trial
    path = os.getcwd()
    
    #Name folder based on percentage
    path = path +  '_Percent_' + str(percent*100);
    
    for i in range(numTrials):
        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(data, data[:,ncolumns], test_size=(1-percent));
        
        #Save out to Excel
        #make directory 
        if not os.path.exists(path):
            os.makedirs(path);
            
        
        #Write training and test data to file 
        np.savetxt(path + '\\' + '\\Sub_select_Trial' + str(i+1) + '_Percent_' + str(percent*100) + '.csv', X_train, delimiter=",");
            
        #Append Trial data and labels
        trial_data.append(X_train);
    
    return trial_data;
