from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1

from sklearn.metrics import confusion_matrix


def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

from sklearn.metrics import roc_auc_score


def get_roc_auc(y_test, y_proba):
    return roc_auc_score(y_test, y_proba)