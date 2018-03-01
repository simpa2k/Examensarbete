import argparse
import os

import matplotlib.pyplot as mpl
import numpy as np

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.datasets import posnett_hindle_devanbu
from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset


def default_pipeline_of(estimator):
    return Pipeline(steps=[('scale', StandardScaler()), ('estimator', estimator)])


estimator_labels = ['mlpc', 'lr', 'nb', 'gnb']
estimators = [
    default_pipeline_of(MLPClassifier()),
    default_pipeline_of(LogisticRegression()),
    MultinomialNB(),
    GaussianNB()
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
    mpl.savefig(output_path + '/curve.png')

    mpl.clf()

    mpl.boxplot(results)
    mpl.savefig(output_path + '/boxplot.png')
    mpl.clf()


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
    k_fold = StratifiedKFold(n_splits=3, random_state=seed)

    return cross_val_score(estimator, X, y, cv=k_fold, scoring='accuracy')


def main():
    parser = setup_parser()
    args = parser.parse_args()

    k_fold_label = args.k_fold_label

    estimator = estimators_by_label[args.estimator]
    dataset = datasets_by_label[args.dataset]
    # X, y = dataset(args.data_root, args.document_directory, args.annotation_directory)
    dataset.load(args.data_root, args.annotation_path)
    dataset.describe(args.output_directory)

    X = dataset.features
    y = np.where(dataset.annotations > 3, 1, 0)

    results = perform_experiment(X, y, estimator)

    print('Mean score was:', results.mean())

    save_results(results, os.path.join(args.output_directory, args.scoring_directory), k_fold_label)

if __name__ == '__main__':
    main()
