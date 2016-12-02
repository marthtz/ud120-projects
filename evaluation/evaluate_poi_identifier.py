#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
#from sklearn.model_selection import train_test_split # 0.18 >
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# Count number of POIs in test set
print sum(x>0 for x in y_test)

# People in test set
print len(y_test)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = accuracy_score(pred, y_test)
print('Accuracy: {:.4f}'.format(score))

# Score if everyone is predicted not POI
score = accuracy_score(pred, [0.]*29)
print('Accuracy: {:.4f}'.format(score))

# Check for true positives in predictions
ii = 0
for index, item in enumerate(pred):
    if item == 1 and item == y_test[index]:
        ii += 1
print ii

# Precision of POI identifier
print precision_score(y_test, pred)

# Recall of POI identifier
print recall_score(y_test, pred)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# Check for true positives in predictions
ii = 0
for index, item in enumerate(predictions):
    if item == 1 and item == true_labels[index]:
        ii += 1
print ii

# Check for true negatives in predictions
ii = 0
for index, item in enumerate(predictions):
    if item == 0 and item == true_labels[index]:
        ii += 1
print ii

# Check for false positives in predictions
ii = 0
for index, item in enumerate(predictions):
    if item == 1 and true_labels[index] == 0:
        ii += 1
print ii

# Check for false negatives in predictions
ii = 0
for index, item in enumerate(predictions):
    if item == 0 and true_labels[index] == 1:
        ii += 1
print ii

# Precision
print precision_score(true_labels, predictions)

# Recall
print recall_score(true_labels, predictions)