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
plt.scatter(bumpy_fast, grade_fast, color = "g", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

#Check out what parameters work well for SVC
parameters = { 'kernel': ['linear', 'rbf', 'poly'], 'C': [1.0, 10.0]};
svr = SVC()
clf1 = GridSearchCV(svr, param_grid=parameters, cv=5)
clf1.fit(features_train, labels_train)
print clf1.score(features_test, labels_test)
print clf1.best_params_

svc_params = { 'kernel': 'poly', 'C': 1.0 }
#Check out how SVC works in AdaBoost
clf = AdaBoostClassifier(SVC(**svc_params), algorithm="SAMME", n_estimators=100)

# scores = cross_val_score(clf, features_train, labels_train)
t0 = time()
clf.fit(features_train, labels_train)
print time() - t0, "seconds to train AdaBoostClassifier"
print clf.score(features_test, labels_test), "AdaBoostClassifier"

#Check out how SVC works on its own with the same parameters
clfPoly = SVC(**svc_params);
t0 = time()
clfPoly.fit(features_train, labels_train)
print time() - t0, "seconds to train SVC with poly kernel"
print clfPoly.score(features_test, labels_test), "SVC with poly kernel"

try:
    prettyPicture(clf, features_test, labels_test)
    prettyPicture(clfPoly, features_test, labels_test)
except NameError:
    print 'What went wrong?'
    pass
