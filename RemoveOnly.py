# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:01:45 2018
@filename:  RemoveOnly.py
@date:      8 Jul 2018
@author:    Joshua Peeples and Diandra Prioleau
"""
import csv
import numpy as np
from numpy import *
import pdb 
import os
from sklearn.neighbors import NearestNeighbors

def RemoveOnly(data,k):
    
    k_prime = (k + 1)/2;
    
    S = data; 

    ncolumns = len(S[1]) - 1;
    
    temp = S[:,1:ncolumns]; #dataset without column labelling glass type
    
    samples_removed = [ ]; #array of samples removed from original training set 
    samples_removed = np.empty((0,len(S[1])),float)
    num_removes = 0; #counts number of samples removed, used as index for samples_removed

    i = 0; #variable to control while loop
    print('temp:',len(temp));
    enter = 0;
    while i < len(S) : #loop based on size of temp
    
        print('i: ', i)
        temp = S[:,1:ncolumns];
        
        #finding k nearest neighbors for each sample in training data
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(temp) 
        distances, indices = nbrs.kneighbors(temp);
        
        count_nlabels = 0;
        temp_indices = indices[i,:];

        for j in range(0,len(temp_indices)):
            index = indices[i,j]; #
            enter += 1;
            if(S[index,ncolumns] == S[i,ncolumns] and index != i):
                count_nlabels += 1;
        
        print(count_nlabels)
        if(count_nlabels < k_prime):
            print('remove');
            print('i before: ', i)

            #append sample before removing for comparison later with AQ outliers
            samples_removed = np.vstack((samples_removed,S[i,:]))
            
            #remove sample from training data
            S= np.delete(S,i,0);
            num_removes += 1;
        else :
            i += 1;
            
            print('i after: ', i)
    
        print('temp:',len(temp));
        print('length of data:', len(S));
    print('# of comparison: ',enter)
    
    
    
    
    #Create directory to save data for each new trial
    path = os.getcwd();
    
    #append trial number 
    path = path + '\\RemovedSamples';
    
    #make directory 
    if not os.path.exists(path):
        os.makedirs(path);
    
    np.savetxt(path + '\\' + 'samples_removed.csv', samples_removed, delimiter=",");

    #return data remaining after executing RemoveOnly algorithm
    return S;
    
