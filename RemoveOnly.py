# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:01:45 2018
@filename:  RemoveOnly.py
@date:      8 Jul 2018
@author:    Joshua Peeples and Diandra Prioleau
"""
import csv
import numpy as np
import pdb 
import os
from sklearn.neighbors import NearestNeighbors

def RemoveOnly(data,k):
    
    k_prime = (k + 1)/2;
    
    S = data; 
    #pdb.set_trace();
    ncolumns = len(S[1]) - 1;
    
    num_removes = 0; #counts number of samples removed 
    temp = S[:,0:ncolumns]; #dataset without column labelling glass type
    
    i = 0; #variable to control while loop
    
    while i < len(temp) - 1: #loop based on size of temp
    
        print('i: ', i)
        temp = S[:,0:ncolumns];
        temp = np.delete(temp,i,0);
        
        #finding k nearest neighbors for each sample in training data
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(temp) 
        distances, indices = nbrs.kneighbors(temp);
        
        count_nlabels = 0;
        temp_indices = indices[i,:];
        #pdb.set_trace();
        for j in range(0,len(temp_indices)):
            index = indices[i,j]; #
            if(S[index,ncolumns] == S[i,ncolumns]):
                count_nlabels += 1;
        
        print(count_nlabels)
        if(count_nlabels < k_prime):
            print('remove');
            print('i before: ', i)
            #remove sample from training data
            S= np.delete(S,i,0);
            num_removes += 1;
        else :
            i += 1;
            
            print('i after: ', i)
    
    #return data remaining after executing RemoveOnly algorithm 
    return S;
    
