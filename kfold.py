# -*- coding: utf-8 -*-
"""
@filename:  10_kfold.py
@date:      8 Jul 2018 
@author:    Joshua Peeples & Diandra Prioleau
"""
import numpy as np
from sklearn.model_selection import KFold
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

k_cv = 10; #10-fold cross validation 
k = 3; #3 nearest neighbors
kf = KFold(n_splits=k_cv, random_state=None, shuffle=True);

#Conduct RemoveOnly
remove_only = True;

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

for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index);
    data_train, data_test = data[train_index], data[test_index]

    #Write training and test data to file 
    np.savetxt(path + '\\' + output_training + str(file_num) + '.csv', data_train, delimiter=",");
    np.savetxt(path + '\\' + output_test + str(file_num) + '.csv', data_test, delimiter=",");
    
   
    if(remove_only):
        #Call RemoveOnly function for each training set
        remove_only_data = RemoveOnly(data_train,k)
        
        #Save data remaining from RemoveOnly 
        np.savetxt(path + '\\' + 'remove_only' + str(file_num) + '.csv', remove_only_data, delimiter=",");
    
    file_num += 1;
