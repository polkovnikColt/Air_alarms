import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def display_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fpr, tpr, _thersholds = metrics.roc_curve(y_test, y_pred)
    cm_list =  confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize = (12,7))
    cm_plot = sns.heatmap(cm_list, annot=True, cmap = "Blues_r")
    cm_plot.set_xlabel("Predicted values")
    cm_plot.set_ylabel("Actual values")
    plt.show()

    report = classification_report(y_test, y_pred, target_names=["Actual", "Pred"])
    print(report)