import argparse
import os
import itertools

import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import posnett_hindle_devanbu
from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset
from src.feature_selection.LoggingFeatureSelector import LoggingFeatureSelector


def default_pipeline_of(estimator):
    return Pipeline(steps=[('scale', StandardScaler()), ('estimator', estimator)])


estimator_labels = ['mlpc', 'lr', 'nb', 'gnb', 'bnb', 'rfc']
estimators = [
    default_pipeline_of(MLPClassifier(random_state=0)),
    default_pipeline_of(LogisticRegression()),
    MultinomialNB(),
    #make_pipeline(LoggingFeatureSelector(SelectFromModel(ExtraTreesClassifier(random_state=0))), GaussianNB()),
    GaussianNB(),
    BernoulliNB(),
    RandomForestClassifier(n_estimators=100, random_state=0)
]
estimators_by_label = dict(zip(estimator_labels, estimators))

dataset_labels = ['phd', 'ldo']
datasets = [posnett_hindle_devanbu.get, LundDighemOlofssonDataset()]
datasets_by_label = dict(zip(dataset_labels, datasets))

cross_validation_column_label = 'cv'
cross_validation_fold_column_label = 'fold'
prediction_column_label = 'pred'

predicted_class_label = 'Class'
document_label = 'Document'

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
    parser.add_argument('--feature-generation',
                        help='Whether to use previously generated features, if they exist, '
                             'or generate new ones regardless of whether computed features exist or not.',
                        default='previous', choices=['previous', 'force'])

    return parser


def save_scores_as_csv(results, output_path, k_fold_label):
    results = pd.DataFrame(results.loc[0])
    results.index = np.append([i for i in range(1, 11)], ['Medelvärde', 'Standardavvikelse'])

    results.index.name = 'Del'
    results = results.rename(columns={0: 'Noggrannhet'})

    results = results.round(3)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results.transpose().to_csv(os.path.join(output_path, 'results.csv'), index=False)
    results.iloc[0:10].to_csv(os.path.join(output_path, 'plottable_results.csv'))
    results['Noggrannhet'].iloc[0:9].to_csv(os.path.join(output_path, 'boxplottable_results.csv'), index=False)
    results.iloc[10:13].transpose().to_csv(os.path.join(output_path, 'errorplottable_results.csv'), index=False)


def save_results(results, output_path, k_fold_label):
    save_scores_as_csv(results, output_path, k_fold_label)


def weighted_accuracy(y_true, y_pred, prediction_gatherer=None):
    if prediction_gatherer is not None:
        prediction_gatherer.append(y_pred)

    """
    From: https://www.quora.com/How-do-you-measure-the-accuracy-score-for-each-class-when-testing-classifier-in-sklearn
    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] *= (i/float(y_true.shape[0]))

    #return accuracy_score(y_true, y_pred, sample_weight=w)
    """

    return accuracy_score(y_true, y_pred)


def create_prediction_dataframe():
    cross_validations = [cross_validation_column_label + str(i) for i in range(0, 10)]
    folds = [cross_validation_fold_column_label + str(i) for i in range(0, 10)]
    pred = [prediction_column_label + str(i) for i in range(0, 8)]

    index = pd.MultiIndex.from_product([cross_validations, folds, pred], names=['Cross validation run',
                                                                                'Cross validation fold',
                                                                                'Prediction'])

    return pd.DataFrame(index=[predicted_class_label, document_label], columns=index)


def perform_experiment(X, y, estimator):
    """
    Runs a 10-fold cross validation, ten times, with a different random seed
    each time, as per (Posnett, Hindle & Devanbu 2011).

    :returns The mean weighted accuracy of each cross validation and the raw predictions mapped
    to the document the prediction was made on.
    """
    columns = np.append([i for i in range(0, 10)], ['mean', 'std'])
    results = pd.DataFrame(columns=columns)
    predictions = create_prediction_dataframe()

    for i in range(0, 10):
        k_fold = StratifiedKFold(n_splits=10, random_state=i)

        predictions_for_this_run = []

        scoring = make_scorer(weighted_accuracy, prediction_gatherer=predictions_for_this_run)

        scores = cross_val_score(estimator, X, y, cv=k_fold, scoring=scoring)
        results = results.append(
            pd.DataFrame(
                [[score for score in np.append(scores, [scores.mean(), scores.std()])]],
                columns=columns),
            ignore_index=True
        )

        record_predictions_for_documents(i, list(k_fold.split(X, y)), predictions, predictions_for_this_run)

    results.loc['mean'] = results.mean()

    return results, predictions


def record_predictions_for_documents(cross_validation_run, splits, data_frame, predictions_for_this_run):
    i = 0
    for fold_predictions in predictions_for_this_run:
        j = 0
        for prediction in fold_predictions:
            column = (cross_validation_column_label + str(cross_validation_run),
                      cross_validation_fold_column_label + str(i),
                      prediction_column_label + str(j))

            data_frame.loc[predicted_class_label, column] = prediction
            data_frame.loc[document_label, column] = splits[i][1][j]
            j = j + 1

        i = i + 1


def save_confusion_matrix(human_to_model_mapping, output_directory, scoring_directory):
    cm = confusion_matrix(human_to_model_mapping['human'], human_to_model_mapping['model'])

    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    columns = ['x', 'y', 'c']
    plottable_matrix_data = pd.concat(
        [pd.DataFrame([[x, y, cm[x, y]]], columns=columns).round(3) for x, y in sorted(itertools.product([0, 1], repeat=2), key=lambda x: x[1])]  # The sorting is needed for pgfplots to be able to draw the matrix plot correctly.
    )

    plottable_matrix_data.to_csv(
        os.path.join(output_directory, scoring_directory, 'plottable_confusion_matrix.csv'),
        index=False
    )


def run(X, y, estimator, annotations_by_document, output_directory, scoring_directory):
    results, predictions = perform_experiment(X, y, estimator)
    human_to_model_mapping = map_predictions_to_annotations(annotations_by_document, predictions)

    print('Mean score was:', results['mean'].mean())

    save_results(results, os.path.join(output_directory, scoring_directory), '')
    save_confusion_matrix(human_to_model_mapping, output_directory, scoring_directory)
    predictions.to_csv(os.path.join(output_directory, scoring_directory, 'predictions.csv'), index_label=False)


def map_predictions_to_annotations(annotations, predictions):
    transposed = predictions.transpose()
    without_nan = transposed.dropna()
    with_dropped_index = without_nan.reset_index()[['Class', 'Document']]
    grouped = with_dropped_index.astype(int).groupby(['Document'])
    with_single_prediction = grouped.mean()  # Currently, predictions are always the same for each document so there will be a single binary classification for each one.

    human_to_model_mapping = annotations.reset_index().join(with_single_prediction).set_index('document')
    human_to_model_mapping.columns = ['human', 'model']

    return human_to_model_mapping


def main():
    parser = setup_parser()
    args = parser.parse_args()

    k_fold_label = args.k_fold_label

    estimator = estimators_by_label[args.estimator]
    dataset = datasets_by_label[args.dataset]

    force_feature_generation = False
    if args.feature_generation == 'force':
        force_feature_generation = True

    dataset.load(args.data_root, args.annotation_path, force_feature_generation)
    dataset.describe(args.output_directory)

    y = dataset.get_annotations()

    run(dataset.get_project_level_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'project_level_features')),
    run(dataset.get_mean_method_level_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'mean_method_level_features')),
    run(dataset.get_project_level_loc_method_level_V_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'project_level_loc_method_level_V')),
    run(dataset.get_method_level_loc_project_level_V_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'method_level_loc_project_level_V')),
    run(dataset.get_all_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'all_features'))
    run(
        dataset.get_all_features(),
        y,
        make_pipeline(LoggingFeatureSelector(SelectFromModel(ExtraTreesClassifier(random_state=0))), estimator),
        dataset.get_binarized_annotations_by_document(),
        args.output_directory,
        os.path.join(args.scoring_directory, 'with_feature_selection')
    )


if __name__ == '__main__':
    main()
