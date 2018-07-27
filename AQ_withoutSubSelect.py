# -*- coding: utf-8 -*-
"""
@filename:  AQ_withoutSubSelect.py
@date:      26 Jul 2018
@author:    Diandra Prioleau
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv
import pdb
import os
from RemoveOnly import RemoveOnly
from Subselect_data import SubSelect
from AQeval import AQeval

#import dataset (Credit: Joshua Peeples, Princess Lyons, Diandra Prioleau)
def ImportData(path):
    raw_data = open(path, 'rt');
    reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
    x = list(reader);
    data = np.array(x).astype('float');
    print(data.shape);
    
    return data;

#kNN classification (Credit: Joshua Peeples, Princess Lyons, Diandra Prioleau)
def Classifier(k,train_data,test_data):
    
    #pdb.set_trace();
    ncolumns = len(train_data[1]) - 1;
    
    #kNN Classifier 
    knn = KNeighborsClassifier(n_neighbors=k);
            
    #fitting the model
    knn.fit(train_data[:,1:ncolumns], train_data[:,ncolumns]);
            
    #predict the response
    pred = knn.predict(test_data[:,1:ncolumns]) ;
    
    return pred;
    
    
#variables
percent_50 = 0.5;
percent_80 = 0.8;
numTrials = 4;
k_neighbors = 3;
k_fold = 10;


AQ_full = ImportData('C:\\Users\\Diandra\\Documents\\HXR Lab\\AQ Project\\Experiment\\Code\\Original Datasets\\recommended_applications_AQ_full.csv');
AQ_test = ImportData('C:\\Users\\Diandra\\Documents\\HXR Lab\\AQ Project\\Experiment\\Code\\Original Datasets\\rejected_applications_AQ_full.csv');

AQeval(AQ_test[:,-1],Classifier(k_neighbors,AQ_full,AQ_test));
 

