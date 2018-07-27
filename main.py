# -*- coding: utf-8 -*-
"""
@filename:      main.py
@date:          22 Jul 2018
@authors:       Princess Lyons, Joshua Peeples, Diandra Prioleau
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

#import dataset
def ImportData(path):
    raw_data = open(path, 'rt');
    reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
    x = list(reader);
    data = np.array(x).astype('float');
    print(data.shape);

    return data;

#kNN classification
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

#Import AQ dataset
AQ_50 = ImportData(os.path.join(os.path.dirname(__file__), 'data_files/recommended-applications_50.csv')) # Relative Paths
AQ_80 = ImportData(os.path.join(os.path.dirname(__file__), 'data_files/recommended-applications_80.csv'))


#TODO: Random 70-30 (training/test) split on dataset - ask Armisha for function that calls random generator
#will be removed later
train_data = ImportData(os.path.join(os.path.dirname(__file__), 'data_files/glass_id_train.csv')) # Relative Paths
test_data = ImportData(os.path.join(os.path.dirname(__file__), 'data_files/glass_id_test.csv'))

#Call SubSelect - returns percentage of training set
train_50 = SubSelect(percent_50,numTrials,train_data);

train_80 = SubSelect(percent_80,numTrials,train_data);

#pdb.set_trace();

for i in range(0,numTrials):
    print('Trial_50',i)
    AQeval(test_data[:,-1],Classifier(k_neighbors,np.array(train_50[i]),test_data));

    print('Trial_80',i)
    AQeval(test_data[:,-1],Classifier(k_neighbors,np.array(train_80[i]),test_data));

print('Trial_AQ_50')
AQeval(test_data[:,-1],Classifier(k_neighbors,AQ_50,test_data));

print('Trial_AQ_80')
AQeval(test_data[:,-1],Classifier(k_neighbors,AQ_80,test_data));
