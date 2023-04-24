import datetime
import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

from utils.metrics_evaluation import display_metrics


param_grid_LR = {"C": [0.1, 1, 100], "penalty": ['l1', 'l2', 'elasticnet', None], 'random_state': [0, 1, 10],
                 'tol': [0.1, 1, 100]}
param_grid_RFC = {'n_estimators': [0, 2, 5, 10],"criterion": ["gini", "entropy", "log_loss"],"max_features": ["sqrt", "log2", None]}

param_grid_MLP = {"activation": ["identity", "logistic", "tanh", "relu"], "solver": ['lbfgs', 'sgd', 'adam'], "alpha": [0.0001, 0.001, 0.05, 0.1],
"max_iter": [200, 300, 400, 500]}

data_path = '/home/vlad/Документы/sparse.npz'
y_path = '/home/vlad/Документы/y.csv'

df_all_data = scipy.sparse.load_npz(data_path)
y = pd.read_csv(y_path)["is_alarm"]

tss = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tss.split(df_all_data):
    X_train, X_test = df_all_data[train_index], df_all_data[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# X_train, X_test, y_train, y_test = train_test_split(df_all_data, y, test_size=0.2, random_state=1, shuffle=True)


# LogisticRegresion
# LR = LogisticRegression()
# LR.fit(X_train, y_train)
# display_metrics(LR, X_test, y_test)
# with open("./data/models/LR.pkl", "wb") as lr:
#     pickle.dump(LR, lr)


# DecisionTree
# NB = DecisionTreeClassifier()
# NB.fit(np.asarray(X_train.todense()), y_train)
# display_metrics(NB, X_test, y_test)
# with open("./data/models/DTC.pkl", "wb") as DTC:
#     pickle.dump(NB, DTC)


# RFC
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# display_metrics(rfc, X_test, y_test)
# with open("./data/models/RFC.pkl", "wb") as RFC:
#     pickle.dump(rfc, RFC)


# #SGD
# sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
# sgd.fit(X_train, y_train)
# display_metrics(sgd, X_test, y_test)
# with open("./data/models/SGD.pkl", "wb") as SGD:
#     pickle.dump(sgd, SGD)


# SVM
# clf = SVC()
# clf.fit(X_train, y_train)
# display_metrics(clf, X_test, y_test)


# Multi-layer Perceptron
# mlp = MLPClassifier()
# mlp.fit(X_train, y_train)
# display_metrics(mlp, X_test, y_test)
# with open("./data/models/MLP.pkl", "wb") as MLP:
#     pickle.dump(mlp, MLP)
