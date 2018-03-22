import argparse
import os

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score

from src.datasets import posnett_hindle_devanbu
from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset
from src.utils.save_data import save_fig, save_as_csv


def default_pipeline_of(estimator):
    return Pipeline(steps=[('scale', StandardScaler()), ('estimator', estimator)])


estimator_labels = ['mlpc', 'lr', 'nb', 'gnb', 'rfc']
estimators = [
    default_pipeline_of(MLPClassifier(random_state=0)),
    default_pipeline_of(LogisticRegression()),
    Pipeline(steps=[('scale', MinMaxScaler()), ('estimator', MultinomialNB())]),
    Pipeline(steps=[('scale', MinMaxScaler()), ('estimator', GaussianNB())]),
    RandomForestClassifier(n_estimators=100, random_state=0)
]
estimators_by_label = dict(zip(estimator_labels, estimators))

dataset_labels = ['phd', 'ldo']
datasets = [posnett_hindle_devanbu.get, LundDighemOlofssonDataset()]
datasets_by_label = dict(zip(dataset_labels, datasets))


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-root', help='Path to the common parent directory of everything to be used as data.',
                        default='../data/bw')
    parser.add_argument('--document-directory',
                        help='Path to the directory containing the documents to be processed, '
                             'relative to the data-root.',
                        default='/snippets')
    parser.add_argument('--annotation-path',
                        help='Path to the correct labels of the documents to be processed, relative to the data-root.',
                        default='/votes.csv')
    parser.add_argument('--output-directory',
                        help='Output directory for data.',
                        default='../output/')
    parser.add_argument('--scoring-directory',
                        help='Output directory for scoring data, relative to the output-directory.',
                        default='scoring')
    parser.add_argument('--k-fold-label', help='A label to put before each k fold iteration in the outputted csv.',
                        default='')
    parser.add_argument('--estimator', help='The machine learning algorithm to be used.',
                        choices=estimator_labels, required=True)
    parser.add_argument('--dataset', help='A function that loads the data to be used.',
                        default='phd', choices=dataset_labels)

    return parser


def save_scores_as_plots(results, output_path):
    mpl.plot(results, label='k_folds')
    mpl.plot(np.full((len(results),), results.mean()), label='mean')
    mpl.legend()
    save_fig(output_path, 'curve.png', mpl)

    mpl.clf()

    mpl.boxplot(results)
    save_fig(output_path, 'boxplot.png', mpl)
    mpl.clf()


def save_scores_as_csv(results, output_path, k_fold_label):
    header = ','.join([k_fold_label + str(i) for i in range(len(results))])
    header += ',mean'

    results = np.append(results, results.mean())

    save_as_csv(
        output_path,
        'results.csv',
        np.array(results)[np.newaxis],
        header
    )


def save_results(results, output_path, k_fold_label):
    save_scores_as_plots(results, output_path)
    save_scores_as_csv(results, output_path, k_fold_label)


def weighted_accuracy(y_true, y_pred, prediction_gatherer=None):
    if prediction_gatherer is not None:
        prediction_gatherer.append(y_pred)

    """
    From: https://www.quora.com/How-do-you-measure-the-accuracy-score-for-each-class-when-testing-classifier-in-sklearn
    """
    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] *= (i/float(y_true.shape[0]))

    return accuracy_score(y_true, y_pred, sample_weight=w)


def create_prediction_dataframe():
    cross_validations = ['cv' + str(i) for i in range(0, 10)]
    folds = ['fold' + str(i) for i in range(0, 10)]
    pred = ['pred' + str(i) for i in range(0, 8)]

    index = pd.MultiIndex.from_product([cross_validations, folds, pred], names=['cv', 'fold', 'p'])

    return pd.DataFrame(index=['pred_class', 'doc'], columns=index)


def perform_experiment(X, y, estimator):
    """
    Runs a 10-fold cross validation, ten times, with a different random seed
    each time, as per (Posnett, Hindle & Devanbu 2011).

    :returns The weighted accuracy of each cross validation fold and the raw predictions mapped
    to the document the prediction was made on.
    """
    results = []
    predictions = create_prediction_dataframe()

    for i in range(0, 9):
        k_fold = StratifiedKFold(n_splits=10, random_state=i)

        predictions_for_this_run = []

        scoring = make_scorer(weighted_accuracy, prediction_gatherer=predictions_for_this_run)

        results.append(
            cross_val_score(estimator, X, y, cv=k_fold, scoring=scoring)
        )

        record_predictions_for_documents(i, list(k_fold.split(X, y)), predictions, predictions_for_this_run)

    return np.array(results).mean(axis=0), predictions


def record_predictions_for_documents(cross_validation_run, splits, data_frame, predictions_for_this_run):
    i = 0
    for fold_predictions in predictions_for_this_run:
        j = 0
        for prediction in fold_predictions:
            column = ('cv' + str(cross_validation_run), 'fold' + str(i), 'pred' + str(j))
            data_frame.loc['pred_class', column] = prediction
            data_frame.loc['doc', column] = splits[i][1][j]
            j = j + 1

        i = i + 1


def main():
    parser = setup_parser()
    args = parser.parse_args()

    k_fold_label = args.k_fold_label

    estimator = estimators_by_label[args.estimator]
    dataset = datasets_by_label[args.dataset]

    dataset.load(args.data_root, args.annotation_path)
    dataset.describe(args.output_directory)

    X = dataset.get_features()
    y = dataset.get_annotations()

    results, predictions = perform_experiment(X, y, estimator)

    print('Mean score was:', results.mean())

    save_results(results, os.path.join(args.output_directory, args.scoring_directory), k_fold_label)
    predictions.to_csv(os.path.join(args.output_directory, args.scoring_directory, 'predictions.csv'), index_label=False)


if __name__ == '__main__':
    main()
