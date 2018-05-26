import os
import itertools

from functools import reduce

import numpy as np

from sklearn.metrics import make_scorer, accuracy_score, recall_score
from sklearn.naive_bayes import GaussianNB

from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset
from src.experiments.experiment import perform, run
from src.utils.save_data import save_common_results_with_and_without_index, save_common_results_without_transpose

data_root = '/home/simon/DSV/KANDIDAT/Uppgifter/'
annotation_path = '../data/ldo/annotations/annotations.csv'
output_directory = '../olofsson_2018/'
scoring_directory = 'scoring/'

dataset = LundDighemOlofssonDataset()

estimator = GaussianNB()
scoring_functions = {
    'accuracy': make_scorer(accuracy_score),
    'tpr': make_scorer(recall_score),
    'tnr': make_scorer(recall_score, pos_label=0)
}


def olofsson_2018():
    perform(run_all_feature_combinations, dataset, data_root, annotation_path, output_directory)


def run_all_feature_combinations(dataset, y, output_directory, scoring_directory):
    results = {}
    only_aggregated_results = gather_results_for_all_feature_combinations(dataset, output_directory, results,
                                                                          scoring_directory, y)

    save_common_results_with_and_without_index(only_aggregated_results,
                                               os.path.join(output_directory, scoring_directory),
                                               'all_aggregated.csv',
                                               separator=';')

    interesting_rows = [0, 4, 23, 28, 29, 30]
    save_common_results_with_and_without_index([only_aggregated_results[i] for i in interesting_rows],
                                               os.path.join(output_directory, scoring_directory),
                                               'selected_aggregated.csv',
                                               separator=';')

    save_results_by_score_label(interesting_rows, output_directory, results, scoring_directory)


def gather_results_for_all_feature_combinations(dataset, output_directory, results,
                                                scoring_directory, y):
    only_aggregated_results = []
    for i in range(len(dataset.all_feature_labels)):
        for combination in itertools.combinations(range(dataset.get_all_features().shape[1]), i + 1):
            latex_feature_names = np.array(['\({}\)'.format(feature_name) for feature_name in
                                            ['R_P', 'V_P', 'E_P', '\overline{R_M}', '\overline{V_M}']])

            joined, mask = run_feature_combination(combination,
                                                   dataset,
                                                   latex_feature_names,
                                                   output_directory,
                                                   results,
                                                   scoring_directory,
                                                   y)

            only_aggregated_results.append(joined[['accuracym', 'accuracystd', 'tprm', 'tprstd', 'tnrm', 'tnrstd']])
            print('Ran combination: {}'.format(mask.astype(int)))

    return only_aggregated_results


def run_feature_combination(combination, dataset, latex_feature_names, output_directory, results, scoring_directory, y):
    X = dataset.get_all_features()[:, list(combination)]
    mask = get_label_mask(combination, dataset)

    result = run(X, y, estimator, scoring_functions,
                 dataset.get_binarized_annotations_by_document(),
                 output_directory,
                 os.path.join(scoring_directory, 'all_combinations',
                              '_'.join(np.array(dataset.all_feature_labels)[mask])))

    only_aggregations = get_aggregations_by_score_label(latex_feature_names, mask, result, results)
    joined = reduce(lambda a, b: a.join(b), only_aggregations)

    return joined, mask


def get_label_mask(combination, dataset):
    mask = np.zeros(len(dataset.all_feature_labels), dtype=bool)
    mask[np.array(combination)] = 1

    return mask


def get_aggregations_by_score_label(latex_feature_names, mask, result, results):
    only_aggregations = []
    for score_label, scoring in result.items():
        scoring = scoring.transpose()
        scoring.index = [', '.join(latex_feature_names[mask])]

        results.setdefault(score_label, []).append(scoring)

        aggregation = scoring[['Medelvärde', 'Standardavvikelse']]
        aggregation.columns = ['{}m'.format(score_label), '{}std'.format(score_label)]
        only_aggregations.append(aggregation)

    return only_aggregations


def save_results_by_score_label(interesting_rows, output_directory, results, scoring_directory):
    for score_label, scores in results.items():
        output_path = os.path.join(output_directory, scoring_directory, score_label)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        save_common_results_without_transpose(scores,
                                              output_path,
                                              'all_results_{}.csv'.format(score_label),
                                              'all_errorplottable_results_{}.csv'.format(score_label),
                                              separator=';')

        save_common_results_without_transpose(
            [scores[i] for i in interesting_rows],
            output_path,
            'results.csv',
            'errorplottable_results.csv',
            separator=';'
        )
