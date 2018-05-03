import numpy as np


def negative_predictive_value(y_true, y_pred):
    true_negatives = np.where(y_pred[y_true == 0] == 0)[0].size
    false_negatives = np.where(y_pred[y_true != 0] == 0)[0].size

    predicted_negatives = true_negatives + false_negatives

    if predicted_negatives == 0:
        return 0
    else:
        return true_negatives / predicted_negatives
