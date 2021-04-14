# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:43:02 2021

@author: tlawrence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from plot_learning_curve import plot_learning_curve
from sklearn.metrics import precision_recall_curve


"""
The file "sonar.mines" contains 111 patterns obtained by bouncing sonar 
signals off a metal cylinder at various angles and under various conditions. 
The file "sonar.rocks" contains 97 patterns obtained from rocks 
under similar conditions. The transmitted sonar signal is a 
frequency-modulated chirp, rising in frequency. The data set contains 
signals obtained from a variety of different aspect angles, 
spanning 90 degrees for the cylinder and 180 degrees for the rock.

Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number 
represents the energy within a particular frequency band, integrated over 
a certain period of time. The integration aperture for higher frequencies 
occur later in time, since these frequencies are transmitted 
later during the chirp.

The label associated with each record contains the letter "R" 
if the object is a rock and "M" if it is a mine (metal cylinder). 
The numbers in the labels are in increasing order of aspect angle, 
but they do not encode the angle directly.

https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

"""
prefix = "FM_Chirp_"
names = [prefix+str(x) for i,x in enumerate(range(60))]
names.append('Target')

df=pd.read_csv('targets.csv')

#X=df.iloc[:,[0]].values
#y=df.iloc[:,[-1]].values

df.columns = names
X = df.drop('Target', axis=1)
y = df['Target']
df.info()
df.describe()

label_encoder = LabelEncoder().fit(y)
y = label_encoder.transform(y)

#df.hist( bins = 60, figsize =(60,40)) 
#plt.show()

#train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
#                                               np.logspace(-7, 3, 3),
#                                               cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)#keep random_state value same to get the same result!!


#clf = svm.SVC(kernel='poly', degree=62)
clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('Precision ' + str(precision_score(y_test, y_pred)))
print('Recall ' + str(recall_score(y_test, y_pred)))
print('Accuracy ' + str(accuracy_score(y_test, y_pred)))
#print(r2_score(y_test, y_pred))

y_scores = clf.decision_function(X_train)
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

#title = r"Learning Curves (SVM, Polynomial kernel, $\deg=62)"
title = r"Learning Curves (SVM, Sigmoid kernel)"
plot_learning_curve(clf, title, X, y)

plt.figure()
df.plot.scatter(x='FM_Chirp_57',y='FM_Chirp_58',c='Target',colormap='viridis')
plt.show()

plt.matshow(cm, cmap = plt.cm.gray) 
plt.show()
