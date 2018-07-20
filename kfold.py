# -*- coding: utf-8 -*-
"""
@filename:  10_kfold.py
@date:      8 Jul 2018
@author:    Joshua Peeples & Diandra Prioleau
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import csv
import pdb
import os
from RemoveOnly import RemoveOnly

#Reads in glass identification from file - file should be training data from random generation
filename = '/Users/princesslyons/Documents/GitHub/AQProject/Sub_select_Trial4_Percent_80.csv';
raw_data = open(filename, 'rt');
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
x = list(reader);
data = np.array(x).astype('float');
print(data.shape);

k_cv = 10; #10-fold cross validation
k = 3; #3 nearest neighbors
kf = KFold(n_splits=k_cv, random_state=None, shuffle=True);

#Conduct RemoveOnly
remove_only = True; #change based on trial

#output filenames for training and test data
output_training = 'kfold_output_training';
output_test = 'kfold_output_test';

trial_num = 1;
file_num = 0;

#Create directory to save data for each new trial
path = os.getcwd();

#append trial number
path = path + '\\Trial' + str(trial_num);

#make directory
if not os.path.exists(path):
    os.makedirs(path);

trial_scores = [];
trial_std = [];
cv_scores = [];

num_trials = 10; #change based on preference

for i in range(0,num_trials):
    for train_index, test_index in kf.split(data):
        print("TRAIN:", train_index, "TEST:", test_index);
        data_train, data_test = data[train_index], data[test_index]

        #Write training and test data to file
        np.savetxt(path + '\\' + output_training + str(file_num) + '.csv', data_train, delimiter=",");
        np.savetxt(path + '\\' + output_test + str(file_num) + '.csv', data_test, delimiter=",");


        np.random.shuffle(data_train);

        if(remove_only):
            #Call RemoveOnly function for each training set
            remove_only_data = RemoveOnly(data_train,k)

            #Save data remaining from RemoveOnly
            np.savetxt(path + '\\' + 'remove_only' + str(file_num) + '.csv', remove_only_data, delimiter=",");

        ncolumns = len(remove_only_data[1]) - 1;

        knn = KNeighborsClassifier(n_neighbors=k);

        # fitting the model
        knn.fit(remove_only_data[:,0:ncolumns], remove_only_data[:,ncolumns]);

        # predict the response
        pred = knn.predict(data_test[:,0:ncolumns]) #remember to change and use with 30% from random generation

        # evaluate accuracy
        score = accuracy_score(data_test[:,ncolumns], pred); #remember to change and use with 30% from random generation
        cv_scores.append(score)
        file_num += 1;
        
        print(score)
        print(confusion_matrix(data_test[:,ncolumns], pred))
        print(classification_report(data_test[:,ncolumns], pred))


    trial_scores.append(np.asarray(cv_scores).mean());
    trial_std.append(np.asarray(cv_scores).std());


print('Trial Mean:', np.asarray(trial_scores).mean())
print('Trial Std: ', np.asarray(trial_std).mean())
