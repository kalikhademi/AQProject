# -*- coding: utf-8 -*-
"""
Filename:   remove_only_alg.py
Date:       08 July 2018
Authors:    Joshua Peeples & Diandra Prioleau
"""
import csv
import numpy as np
import pdb 
from sklearn.neighbors import NearestNeighbors

#Reads in glass identification from file 
filename = 'glass_id_original.csv';
raw_data = open(filename, 'rt');
reader = csv.reader(raw_data, delimiter=',',quoting=csv.QUOTE_NONE);
x = list(reader);
data = np.array(x).astype('float');
print(data.shape);

#k has been chosen to be 3 and k' to be 2 based on previous literature 
k = 3;
k_prime = 2;

S = data; 
ncolumns = len(S[1]) - 1;

num_removes = 0; #counts number of samples removed 
temp = S[:,0:ncolumns-1]; #dataset without column labelling glass type

i = 0; #variable to control while loop

while i < len(temp): #loop based on size of temp

    print('i: ', i)
    temp = S[:,0:ncolumns-1];
    temp = np.delete(temp,i,0);
    
    #finding k nearest neighbors for each sample in training data
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(temp) 
    distances, indices = nbrs.kneighbors(temp);
    
    count_nlabels = 0;
    temp_indices = indices[i,:];
    
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

        

