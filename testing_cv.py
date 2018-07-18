# -*- coding: utf-8 -*-
"""
@filename:      testing_cv.py
@description:   built-in function for conducting k-fold cross validation and classifier 
@author:           Joshua Peeples & Diandra Prioleau
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
filename = 'C:\\Users\\Diandra\\Documents\\HXR Lab\\AQ Project\\Experiment\Original Datasets\\glass_id_original.csv';
raw_data = open(filename, 'rt');
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
x = list(reader);
data = np.array(x).astype('float');
print(data.shape);

ncolumns = len(data[1]) - 1;

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(data[:,0:ncolumns], data[:,ncolumns], test_size=0.3)

#pdb.set_trace();
k_cv = 10; #10-fold cross validation 
k = 3; #3 nearest neighbors
cv_scores = [];

for j in range(0,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    #pdb.set_trace();
    #put back np.random.shuffle(data);
    np.random.shuffle(data);
    #dataset = np.c_[X_train,y_train];
    #dataset = np.c_[X_train,y_train];#
    dataset = data;
    #pdb.set_trace();
    S = RemoveOnly(dataset,k);
    #pdb.set_trace()
    scores = cross_val_score(knn, S[:,0:ncolumns], S[:,ncolumns], cv=k_cv, scoring='accuracy')
    cv_scores.append(scores.mean())

print("Mean: ",np.asarray(cv_scores).mean())
print("Std. Dev.: ", np.asarray(cv_scores).std())
