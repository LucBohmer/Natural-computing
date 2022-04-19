import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#load and split data
data,target = load_wine(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=42)
Acc_ensemble = []

for i in range(1,20):
    #AdaBoost with different sizes of estimators
    clf = AdaBoostClassifier(n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    Acc_ensemble.append(clf.score(X_test, y_test))

plt.plot(Acc_ensemble)
plt.show()

#baseline of only 1 estimator
clf = AdaBoostClassifier(n_estimators=1, random_state=0)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print("Baseline accuracy  of 1 estimator is:"+ "{}".format(score))

Acc_ensemble = []
for i in range(1,30):
    #AdaBoost with different sizes of max depth
    clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=i), n_estimators=1, random_state=0)
    clf.fit(X_train, y_train)
    Acc_ensemble.append(clf.score(X_test, y_test))

plt.plot(Acc_ensemble)
plt.show()

Acc_ensemble = []
for i in range(1,30):
    #AdaBoost with different parameters
    clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1,splitter="random"),n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    Acc_ensemble.append(clf.score(X_test, y_test))

plt.plot(Acc_ensemble)
plt.show()