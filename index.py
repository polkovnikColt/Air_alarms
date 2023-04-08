import datetime
import numpy as np
import pandas as pd
import os
import pathlib
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

df_all_data = scipy.sparse.load_npz('/home/pasha/Study/API_Design/Air_alarms/sparse.npz')[:1000] 
y = pd.read_csv('/home/pasha/Study/API_Design/Air_alarms/y.csv')["is_alarm"][:1000] 

X_train, X_test, y_train, y_test = train_test_split(df_all_data, y, test_size=0.2, random_state=1, shuffle=True)

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

report = classification_report(y_test, Y_pred_NB, target_names=["Actual", "Pred"])
print(report)
