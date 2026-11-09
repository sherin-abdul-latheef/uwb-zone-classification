import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

def evaluate(y_test, predictions, zones=(1, 2, 3, 4)):

    mcm = multilabel_confusion_matrix(y_test, predictions, labels=zones)
    accuracy_per_zone = np.array([
        (cm[1, 1] + cm[0, 0]) / cm.sum()   # (TP + TN) / total
        for cm in mcm
    ])
    return {
        "accuracy"  : accuracy_score(y_test, predictions),
        "confusion" : confusion_matrix(y_test, predictions),
        "precision" : precision_score(y_test, predictions, average='weighted'),
        "recall"    : recall_score(y_test, predictions, average='weighted'),
        "f1"        : f1_score(y_test, predictions, average='weighted'),

        # Per-zone breakdown (average=None keeps one value per zone, not collapsed)
        "accuracy_per_zone" : accuracy_per_zone,
        "precision_per_zone": precision_score(y_test, predictions, labels=zones, average=None, zero_division=0),
        "recall_per_zone"   : recall_score(y_test, predictions, labels=zones, average=None, zero_division=0),
        "f1_per_zone"       : f1_score(y_test, predictions, labels=zones, average=None, zero_division=0),
        "zone_labels"       : list(zones),
    }