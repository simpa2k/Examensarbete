import argparse
import os
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import posnett_hindle_devanbu
from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset
from src.feature_selection.LoggingFeatureSelector import LoggingFeatureSelector
from src.scoring.negative_predictive_value import negative_predictive_value
from src.scoring.plot_roc_curve import plot_cv_roc_curve, plot_final_roc_curve


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
    for score_label, scoring in results.items():
        scoring = pd.DataFrame(scoring.loc[0])
        scoring.index = np.append([i for i in range(1, 11)], ['Medelvärde', 'Standardavvikelse'])

        scoring.index.name = 'Del'
        scoring = scoring.rename(columns={0: score_label})

        scoring = scoring.round(3)

        score_directory = os.path.join(output_path, score_label)
        if not os.path.exists(score_directory):
            os.makedirs(score_directory)

        scoring.transpose().to_csv(os.path.join(score_directory, 'results.csv'), index=False)
        scoring.iloc[0:10].to_csv(os.path.join(score_directory, 'plottable_results.csv'))
        scoring[score_label].iloc[0:9].to_csv(os.path.join(score_directory, 'boxplottable_results.csv'), index=False)
        scoring.iloc[10:13].transpose().to_csv(os.path.join(score_directory, 'errorplottable_results.csv'), index=False)

        results[score_label] = scoring

    return results


def save_results(results, output_path, k_fold_label):
    return save_scores_as_csv(results, output_path, k_fold_label)


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

    results = {
        'accuracy': pd.DataFrame(columns=columns),
        'precision': pd.DataFrame(columns=columns),
        'npv': pd.DataFrame(columns=columns),
        'roc_auc': pd.DataFrame(columns=columns)
    }

    predictions = create_prediction_dataframe()

    for i in range(0, 10):
        k_fold = StratifiedKFold(n_splits=10, random_state=i)

        predictions_for_this_run = []

        #scoring = make_scorer(weighted_accuracy, prediction_gatherer=predictions_for_this_run)
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'npv': make_scorer(negative_predictive_value),
            'roc_auc': make_scorer(plot_cv_roc_curve, needs_proba=True)
        }

        #scores = cross_val_score(estimator, X, y, cv=k_fold, scoring=scoring)
        scores = cross_validate(estimator, X, y, cv=k_fold, scoring=scoring)

        results['accuracy'] = results['accuracy'].append(
            pd.DataFrame(
                [[score for score in np.append(scores['test_accuracy'], [scores['test_accuracy'].mean(), scores['test_accuracy'].std()])]],
                columns=columns),
            ignore_index=True
        )

        results['precision'] = results['precision'].append(
            pd.DataFrame(
                [[score for score in np.append(scores['test_precision'], [scores['test_precision'].mean(), scores['test_precision'].std()])]],
                columns=columns),
            ignore_index=True
        )

        results['npv'] = results['npv'].append(
            pd.DataFrame(
                [[score for score in np.append(scores['test_npv'], [scores['test_npv'].mean(), scores['test_npv'].std()])]],
                columns=columns),
            ignore_index=True
        )

        results['roc_auc'] = results['roc_auc'].append(
            pd.DataFrame(
                [[score for score in np.append(scores['test_roc_auc'], [scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()])]],
                columns=columns),
            ignore_index=True
        )

        record_predictions_for_documents(i, list(k_fold.split(X, y)), predictions, predictions_for_this_run)

    plot_final_roc_curve()

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

    np.savetxt(
        os.path.join(output_directory, scoring_directory, 'confusion_matrix.csv'),
        cm,
        delimiter=',',
        fmt='%5.0f'
    )

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
    #human_to_model_mapping = map_predictions_to_annotations(annotations_by_document, predictions)

    #print('Mean accuracy score was:', results['accuracy']['mean'].mean())
    #print('Mean precision score was:', results['precision']['mean'].mean())
    for score_label, scoring in results.items():
        print('Mean {} score was: {}'.format(score_label, scoring['mean'].mean()))

    formatted_results = save_results(results, os.path.join(output_directory, scoring_directory), '')
    #save_confusion_matrix(human_to_model_mapping, output_directory, scoring_directory)
    predictions.to_csv(os.path.join(output_directory, scoring_directory, 'predictions.csv'), index_label=False)

    return formatted_results


def map_predictions_to_annotations(annotations, predictions):
    transposed = predictions.transpose()
    without_nan = transposed.dropna()
    with_dropped_index = without_nan.reset_index()[['Class', 'Document']]
    grouped = with_dropped_index.astype(int).groupby(['Document'])
    with_single_prediction = grouped.mean()  # Currently, predictions are always the same for each document so there will be a single binary classification for each one.

    human_to_model_mapping = annotations.reset_index().join(with_single_prediction).set_index('document')
    human_to_model_mapping.columns = ['human', 'model']

    return human_to_model_mapping


def save_common_results(scoring_dicts, output_directory):
    save_common_results_without_transpose([scoring_dict['accuracy'].transpose() for scoring_dict in scoring_dicts], output_directory, 'results.csv', 'errorplottable_results.csv')
    """
    concatenated = pd.concat([data_frame.transpose() for data_frame in data_frames]) # All CV results are the same anyway, just choose the first one.
    concatenated = concatenated\
        .reset_index()\
        .drop('index', axis=1)

    concatenated.index.name = 'x'

    concatenated.to_csv(os.path.join(output_directory, 'results.csv'))

    errorplottable = concatenated.copy()[['Medelvärde', 'Standardavvikelse']]
    errorplottable.columns = ['mean', 'std']

    errorplottable.to_csv(os.path.join(output_directory, 'errorplottable_results.csv'))
    """


def save_common_results_without_transpose(data_frames, output_directory, results_filename, errorplottable_results_filename):
    concatenated = pd.concat([data_frame for data_frame in data_frames], ignore_index=True) # All CV results are the same anyway, just choose the first one.
    #concatenated = concatenated \
    #    .reset_index() \
    #    .drop('index', axis=1)

    concatenated.index.name = 'x'

    concatenated.to_csv(os.path.join(output_directory, results_filename))

    errorplottable = concatenated.copy()[['Medelvärde', 'Standardavvikelse']]
    errorplottable.columns = ['mean', 'std']

    errorplottable.to_csv(os.path.join(output_directory, errorplottable_results_filename))


def save_feature_selection_results(output_directory):
    feature_names = np.array(['\({}\)'.format(feature_name) for feature_name in ['R_P', 'V_P', 'E_P', '\overline{R_M}', '\overline{V_M}']])

    feature_counts = pd.DataFrame.from_dict(LoggingFeatureSelector.selected_feature_counts, orient='index')
    feature_counts.columns = ['Antal']
    feature_counts.index = [' '.join(feature_name for feature_name in feature_names[np.array(mask)]) for mask in
                            list(feature_counts.index)]
    feature_counts.index.name = 'Egenskapsuppsättning'

    features_by_first_cv = pd.DataFrame([' '.join(feature_name for feature_name in feature_names[np.array(mask)]) for mask in LoggingFeatureSelector.selected_features_by_cv[0]]).transpose()
    features_by_first_cv.columns = [str(i) for i in range(1, 11)]

    mean_feature_importances_by_cv = [np.mean(importances, axis=0) for importances in LoggingFeatureSelector.feature_importance_by_cv.values()]
    print(mean_feature_importances_by_cv)

    output_path = os.path.join(output_directory, 'feature_selection')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    feature_counts.to_csv(os.path.join(output_path, 'featureset_counts.csv'))
    features_by_first_cv.to_csv(os.path.join(output_path, 'features_by_cv.csv'), index=False)


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

    """
    results = {}
    for i in range(len(dataset.all_feature_labels)):
        for combination in itertools.combinations(range(dataset.get_all_features().shape[1]), i + 1):
            latex_feature_names = np.array(['\({}\)'.format(feature_name) for feature_name in ['R_P', 'V_P', 'E_P', '\overline{R_M}', '\overline{V_M}']])

            X = dataset.get_all_features()[:, list(combination)]

            mask = np.zeros(len(dataset.all_feature_labels), dtype=bool)
            mask[np.array(combination)] = 1
            string_mask = ''.join(map(str, mask))

            result = run(X, y, estimator,
                         dataset.get_binarized_annotations_by_document(),
                         args.output_directory,
                         os.path.join(args.scoring_directory, 'all_combinations', '_'.join(np.array(dataset.all_feature_labels)[mask])))

            for score_label, scoring in result.items():
                scoring = scoring.transpose()
                scoring = scoring.reset_index().drop('index', axis=1)
                #scoring.index = [''.join(latex_feature_names[mask])]
                #scoring.index = [''.join(map(str, mask.astype(int)))]

                results.setdefault(score_label, []).append(scoring)

            print('Ran combination: {}'.format(mask.astype(int)))

    for score_label, scores in results.items():
        output_path = os.path.join(args.output_directory, args.scoring_directory, score_label)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        save_common_results_without_transpose(scores, output_path, 'all_results.csv', 'all_errorplottable_results.csv')

        interesting_rows = [3, 15, 23, 24, 27, 30]
        save_common_results_without_transpose(
            [scores[i] for i in interesting_rows],  # Creating a Numpy array and indexing with the list directly isn't working for some reason.
            output_path,
            'results.csv',
            'errorplottable_results.csv'
        )

    """
    #selection_pipeline = make_pipeline(LoggingFeatureSelector(RFECV(ExtraTreesClassifier(random_state=0), scoring='accuracy')), estimator)

    results = [
        run(dataset.get_project_level_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'project_level_features')),
        run(dataset.get_mean_method_level_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'mean_method_level_features')),
        run(dataset.get_project_level_loc_method_level_V_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'project_level_loc_method_level_V')),
        run(dataset.get_method_level_loc_project_level_V_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'method_level_loc_project_level_V')),
        run(dataset.get_all_features(), y, estimator, dataset.get_binarized_annotations_by_document(), args.output_directory, os.path.join(args.scoring_directory, 'all_features')),
        """
        run(
            dataset.get_all_features(),
            y,
            #make_pipeline(LoggingFeatureSelector(SelectFromModel(ExtraTreesClassifier(random_state=0))), estimator),
            selection_pipeline,
            dataset.get_binarized_annotations_by_document(),
            args.output_directory,
            os.path.join(args.scoring_directory, 'with_feature_selection')
        )
        """
    ]

    save_common_results(results, os.path.join(args.output_directory, args.scoring_directory))
    save_feature_selection_results(args.output_directory)


if __name__ == '__main__':
    main()
