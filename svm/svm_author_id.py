#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

#clf = svm.SVC(kernel='linear')   # 98% with all training, 88% with 1% traingin
#clf = svm.SVC(kernel='rbf', C=10.0)   # 61% with 1% training
#clf = svm.SVC(kernel='rbf', C=100.0)   # 61% with 1% training
#clf = svm.SVC(kernel='rbf', C=1000.0)   # 82% with 1% training
clf = svm.SVC(kernel='rbf', C=10000.0)   # 89% with 1% training

# Set training set to 1% of original data
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 


clf.fit(features_train, labels_train)
t0 = time()
pred = clf.predict(features_test)
#print pred[10]
#print pred[26]
#print pred[50]

uni, counts = np.unique(pred, return_counts=True)

print uni
print counts

print "predict time:", round(time()-t0, 3), "s"

score = accuracy_score(pred, labels_test)

print("Accuracy: {}".format(score))