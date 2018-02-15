import argparse
import os

import matplotlib.pyplot as mpl
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from src.datasets import posnett_hindle_devanbu


def default_pipeline_of(estimator):
    return Pipeline(steps=[('scale', StandardScaler()), ('estimator', estimator)])


estimator_labels = ['mlpc', 'lr', 'nb']
estimators = [
    default_pipeline_of(MLPClassifier()),
    default_pipeline_of(LogisticRegression()),
    MultinomialNB()
]
estimators_by_label = dict(zip(estimator_labels, estimators))

dataset_labels = ['phd']
datasets = [posnett_hindle_devanbu.get]
datasets_by_label = dict(zip(dataset_labels, datasets))


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-root', help='Path to the common parent directory of everything to be used as data.',
                        default='../data/bw')
    parser.add_argument('--document-directory',
                        help='Path to the directory containing the documents to be processed, '
                             'relative to the data-root.',
                        default='/snippets')
    parser.add_argument('--annotation-directory',
                        help='Path to the correct labels of the documents to be processed, relative to the data-root.',
                        default='/votes.csv')
    parser.add_argument('--output-directory',
                        help='Output directory for scoring data.',
                        default='../output/scoring')
    parser.add_argument('--k-fold-label', help='A label to put before each k fold iteration in the outputted csv.',
                        default='')
    parser.add_argument('--estimator', help='The machine learning algorithm to be used.',
                        choices=estimator_labels, required=True)
    parser.add_argument('--dataset', help='A function that loads the data to be used.',
                        default='phd', choices=dataset_labels)

    return parser


def save_scores_as_plots(results, output_path):
    mpl.plot(results)
    mpl.savefig(output_path + '/curve.png')
    mpl.boxplot(results)
    mpl.savefig(output_path + '/boxplot.png')


def save_scores_as_csv(results, output_path, k_fold_label):
    header = ','.join([k_fold_label + str(i) for i in range(len(results))])
    header += ',mean'

    results = np.append(results, results.mean())

    np.savetxt(os.path.join(output_path, 'results.csv'),
               np.array(results)[np.newaxis],
               header=header,
               comments='',
               delimiter=',',
               fmt='%5.2f')


def save_results(results, output_path, k_fold_label):
    save_scores_as_plots(results, output_path)
    save_scores_as_csv(results, output_path, k_fold_label)


def perform_experiment(X, y, estimator):
    seed = 0
    k_fold = StratifiedKFold(n_splits=10, random_state=seed)

    return cross_val_score(estimator, X, y, cv=k_fold, scoring='accuracy')


def main():
    parser = setup_parser()
    args = parser.parse_args()

    k_fold_label = args.k_fold_label

    estimator = estimators_by_label[args.estimator]
    dataset = datasets_by_label[args.dataset]
    X, y = dataset(args.data_root, args.document_directory, args.annotation_directory)

    results = perform_experiment(X, y, estimator)
    save_results(results, args.output_directory, k_fold_label)


if __name__ == '__main__':
    main()
