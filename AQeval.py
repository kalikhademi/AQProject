#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:56:11 2018

@author: princesslyons
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def AQeval(true, predicted):
    
        print(confusion_matrix(true, predicted))
        print(classification_report(true, predicted))    
        print('Accuracy Score: ', accuracy_score(true,predicted))
#        print(precision_recall_fscore_support(true, predicted))
    
    
"""    
Accuracy_ = TN/ TN+FP
Accuracy+ (Recall) = TP/ FN+TP
Weighted Accuracy = 𝞪(Accuracy+) + (1 -𝞪) (Accuracy_) (P.S. 𝞪 is the learning rate and it is between 0 to 1)
Precision = TP/TP+FP
F_measure = 2* (precision * Accuracy+  / precision + Accuracy+ )
"""
