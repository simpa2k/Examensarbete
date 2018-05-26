import os

from sklearn.metrics import make_scorer, accuracy_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.ldo.LundDighemOlofssonDataset import LundDighemOlofssonDataset
from src.experiments.experiment import perform, run
from src.utils.save_data import save_common_results_without_transpose

data_root = '/home/simon/DSV/KANDIDAT/Uppgifter/'
annotation_path = '../data/ldo/annotations/annotations.csv'
output_directory = '../dighem_lund_2018/'
scoring_directory = 'scoring/'

dataset = LundDighemOlofssonDataset()

estimator = make_pipeline(StandardScaler(), MLPClassifier())
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'tpr': make_scorer(recall_score),
    'tnr': make_scorer(recall_score, pos_label=0)
}


def dighem_lund_2018():
    perform(run_improved_halstead_model, dataset, data_root, annotation_path, output_directory)


def run_improved_halstead_model(dataset, y, output_directory, scoring_directory):
    feature_set_labels = [
        'Projektnivå',
        'Metodnivå',
        'Rader kod projektnivå Halsteads V medelvärde metod',
        'Rader kod medelvärde metod Halsteads V projektnivå',
        'Alla egenskaper'
    ]

    results = gather_results(dataset, output_directory, scoring_directory, y)
    results_by_scoring = get_results_by_score_label(feature_set_labels, results)

    save_results_by_score_label(output_directory, results_by_scoring, scoring_directory)


def save_results_by_score_label(output_directory, results_by_scoring, scoring_directory):
    for score_label, scores in results_by_scoring.items():
        save_common_results_without_transpose(
            scores,
            os.path.join(output_directory, scoring_directory),
            '{}_results.csv'.format(score_label),
            '{}_errorplottable_results.csv'.format(score_label),
            ignore_index=True
        )


def gather_results(dataset, output_directory, scoring_directory, y):
    results = [
        run(dataset.get_project_level_features(),
            y,
            estimator,
            scoring,
            dataset.get_binarized_annotations_by_document(),
            output_directory, os.path.join(scoring_directory, 'project_level_features')),
        run(dataset.get_mean_method_level_features(),
            y,
            estimator,
            scoring,
            dataset.get_binarized_annotations_by_document(),
            output_directory,
            os.path.join(scoring_directory, 'mean_method_level_features')),
        run(dataset.get_project_level_loc_method_level_V_features(),
            y,
            estimator,
            scoring,
            dataset.get_binarized_annotations_by_document(),
            output_directory,
            os.path.join(scoring_directory, 'project_level_loc_method_level_V')),
        run(dataset.get_method_level_loc_project_level_V_features(),
            y,
            estimator,
            scoring,
            dataset.get_binarized_annotations_by_document(),
            output_directory,
            os.path.join(scoring_directory, 'method_level_loc_project_level_V')),
        run(dataset.get_all_features(),
            y,
            estimator,
            scoring,
            dataset.get_binarized_annotations_by_document(),
            output_directory, os.path.join(scoring_directory, 'all_features')),
    ]

    return results


def get_results_by_score_label(feature_set_labels, results):
    results_by_score_label = {}
    for feature_set_label, scoring_dict in zip(feature_set_labels, results):
        for score_label, score in scoring_dict.items():
            score = score.transpose()
            score = score.rename({score_label: feature_set_label})
            results_by_score_label.setdefault(score_label, []).append(score)

    return results_by_score_label


