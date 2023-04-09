import datetime
import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

param_grid_LR = {"C": [0.1, 1, 100], "penalty": ['l1', 'l2', 'elasticnet', None], 'random_state': [0,1,10], 
                 'tol': [0.1, 1, 100]}
                 
data_path = ''    
y_path = ''             

df_all_data = scipy.sparse.load_npz(data_path)
y = pd.read_csv(y_path)["is_alarm"]

tss = TimeSeriesSplit(n_splits = 5)
for train_index, test_index in tss.split(df_all_data):
    X_train, X_test = df_all_data[train_index], df_all_data[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#X_train, X_test, y_train, y_test = train_test_split(df_all_data, y, test_size=0.2, random_state=1, shuffle=True)

#LogisticRegresion
LR = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 3,n_jobs=-1)
LR.fit(X_train, y_train)
filename = 'finalized_model_LR.sav'
pickle.dump(LR, open(filename, 'wb'))
Y_pred = LR.predict(X_test)
accuracy = metrics.accuracy_score(y_test, Y_pred)
fpr, tpr, _thersholds = metrics.roc_curve(y_test, Y_pred)
auc_list = round(metrics.auc(fpr, tpr),2)
cm_list =  confusion_matrix(y_test, Y_pred)

fig = plt.figure(figsize = (15,15))
sub = fig.add_subplot(2,3,1).set_title("LogisticRegression")
cm_plot = sns.heatmap(cm_list, annot=True, cmap = "Blues_r")
cm_plot.set_xlabel("Predicted values")
cm_plot.set_ylabel("Actual values")
cm_plot.show()
fig.show()

report = classification_report(y_test, Y_pred, target_names=["Actual", "Pred"])
print(report)


#DecisionTree
NB = DecisionTreeClassifier()
NB.fit(np.asarray(X_train.todense()), y_train)
Y_pred_NB = NB.predict(X_test)
accuracy_NB = metrics.accuracy_score(y_test, Y_pred_NB)
fpr, tpr, _thersholds = metrics.roc_curve(y_test, Y_pred_NB)
auc_list_NB = round(metrics.auc(fpr, tpr),2)
cm_list_NB =  confusion_matrix(y_test, Y_pred_NB)

fig1 = plt.figure(figsize = (15,15))
sub = fig1.add_subplot(2,3,1).set_title("GaussianNB")
cm_plot1 = sns.heatmap(cm_list_NB, annot=True, cmap = "Blues_r")
cm_plot1.set_xlabel("Predicted values")
cm_plot1.set_ylabel("Actual values")
cm_plot1.show()
fig1.show()

report = classification_report(y_test, Y_pred_NB, target_names=["Actual", "Pred"])
print(report)


#SVM
clf = SVC()
clf.fit(X_train, y_train)
Y_pred_svm = clf.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test, Y_pred_svm)
fpr, tpr, _thersholds = metrics.roc_curve(y_test, Y_pred_svm)
auc_list_svm = round(metrics.auc(fpr, tpr),2)
cm_list_svm =  confusion_matrix(y_test, Y_pred_svm)

fig2 = plt.figure(figsize = (15,15))
sub = fig2.add_subplot(2,3,1).set_title("SVM")
cm_plot2 = sns.heatmap(cm_list_svm, annot=True, cmap = "Blues_r")
cm_plot2.set_xlabel("Predicted values")
cm_plot2.set_ylabel("Actual values")
cm_plot2.show()
fig2.show()

report = classification_report(y_test, Y_pred_svm, target_names=["Actual", "Pred"])
print(report)

with open("./data/models/SVM.pkl", "wb") as SVM:
    pickle.dump(clf, SVM)


