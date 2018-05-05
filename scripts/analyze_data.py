import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.plot_confusion_matrix import plot_confusion_matrix

scoring_directory = 'output/scoring/'

index_col = 'x'
mean_col = 'Medelvärde'
std_col = 'Standardavvikelse'

correct_datatypes = {
    index_col: 'str',
    '1': 'float',
    '2': 'float',
    '3': 'float',
    '4': 'float',
    '5': 'float',
    '6': 'float',
    '7': 'float',
    '8': 'float',
    '9': 'float',
    '10': 'float',
    mean_col: 'float',
    std_col: 'float'
}


def path_to_results(score_label):
    return os.path.join(scoring_directory, score_label, 'all_results_{}.csv'.format(score_label))


def read_confusion_matrix(*feature_labels):
    return np.genfromtxt(path_to_confusion_matrix(feature_labels), delimiter=',')


def path_to_confusion_matrix(feature_labels):
    return os.path.join(
        scoring_directory,
        'all_combinations',
        join_feature_labels(feature_labels),
        'confusion_matrix.csv'
    )


def join_feature_labels(feature_labels):
    return ''.join(feature_labels)


def get_score_col(score, score_label):
    return '{}{}'.format(score, score_label)


def read_scoring_df(score_label):
    df = pd.read_csv(path_to_results(score_label), dtype=correct_datatypes)[[index_col, mean_col, std_col]]
    df.columns = [
        index_col,
        get_score_col(score_label, mean_label),
        get_score_col(score_label, std_label),
    ]

    return df


accuracy_label = 'accuracy'
precision_label = 'ppv'
npv_label = 'npv'
roc_auc_label = 'roc_auc'
tpr_label = 'tpr'
tnr_label = 'tnr'

mean_label = 'm'
std_label = 'std'

acc_df = read_scoring_df(accuracy_label)
prec_df = read_scoring_df(precision_label)
npv_df = read_scoring_df(npv_label)
tpr_df = read_scoring_df(tpr_label)
tnr_df = read_scoring_df(tnr_label)

rows = [0, 23, 28, 29, 30]

merged = acc_df\
    .merge(prec_df, on=index_col, how='left')\
    .merge(npv_df, on=index_col, how='left')\
    .merge(tpr_df, on=index_col, how='left')\
    .merge(tnr_df, on=index_col, how='left')

# Plot accuracy, precision and negative predictive value for all feature sets
merged[[
    get_score_col(accuracy_label, mean_label),
    get_score_col(precision_label, mean_label),
    get_score_col(npv_label, mean_label),
]].plot()
plt.show()

# Plot accuracy, precision and negative predictive value for selected rows
merged[[get_score_col(accuracy_label, mean_label), get_score_col(precision_label, mean_label), get_score_col(npv_label, mean_label)]].iloc[rows].plot()
plt.show()

# Error bars for all feature sets
plt.errorbar(
    x=list(merged.index),
    y=merged[get_score_col(accuracy_label, mean_label)],
    yerr=merged[get_score_col(accuracy_label, std_label)],
    fmt='o'
)
plt.show()

# Error bars for selected rows
plt.errorbar(
    x=rows,
    y=merged[get_score_col(accuracy_label, mean_label)].iloc[rows],
    yerr=merged[get_score_col(accuracy_label, std_label)].iloc[rows],
    fmt='o'
)
plt.show()

# Confusion matrices for selected rows


def confusion_matrix_for_row(row):
    npv = merged[get_score_col(npv_label, mean_label)].iloc[row]
    ppv = merged[get_score_col(precision_label, std_label)].iloc[row]

    return np.array([[npv, 1 - ppv], [1 - npv, ppv]])


rows_and_matrices = [
    (
        '{} (acc={:.2f})'.format(row, merged[get_score_col(accuracy_label, mean_label)].iloc[row]),
        confusion_matrix_for_row(row)
    ) for row in rows
]
for row, cm in rows_and_matrices:
    plot_confusion_matrix(cm, ['-', '+'], title=row)
    plt.show()

