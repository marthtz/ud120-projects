#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
#plt.draw()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Gauss NB
##################################
clf = GaussianNB()

# SVM
##################################
c = 50000.0
kernel = 'rbf'
clf = svm.SVC(kernel=kernel, C=c)

# Decission Tree
##################################
min_split = 10
clf = tree.DecisionTreeClassifier(min_samples_split=min_split, max_depth=3)

# KNN k-nearest neighbors
##################################
n_neighbors = 10
weights = 'distance' #'uniform' #'distance'
algo = 'brute'
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algo)

# Random Forest
##################################
n_est = 5
max_depth = None
min_split = 50
max_features = 'log2'
clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, min_samples_split=min_split, max_features=max_features)

# AdaBoost
##################################
n_est = 100
learning_rate = 0.7
algorithm  = 'SAMME'
clf = AdaBoostClassifier(n_estimators=n_est, learning_rate=learning_rate, algorithm=algorithm)

# Fitting
##################################
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
score = accuracy_score(pred, labels_test)

print('Accuracy NB: {:.4f}'.format(score))

# Prints
##################################
text = 'GaussNB_{:.4f}'.format(score)
text = 'SVM_{:.4f}_{}_{}'.format(score, kernel, c)
text = 'Decission Tree_{:.4f}_minsplit{}'.format(score, min_split)
text = 'KNN_{:.4f}_neigh{}_weights{}_algo{}'.format(score, n_neighbors, weights, algo)
text = 'RandomForest_{:.4f}_nest{}_minsplit{}_depth{}_feat{}'.format(score, n_est, min_split, max_depth, max_features )
text = 'AdaBoost_{:.4f}_nest{}_learnrate{}_algo{}'.format(score, n_est, learning_rate, algorithm)

try:
    prettyPicture(clf, features_test, labels_test, text)
except NameError:
    pass
