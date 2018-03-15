from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau, pearsonr, spearmanr
from nltk.metrics import agreement

cohens_kappa_label = 'cohens_kappa'
weighted_cohens_kappa_label = 'weighted_cohens_kappa'
kendalls_tau_label = 'kendalls_tau'
pearsons_r_label = 'pearsons_r'
spearmans_rho_label = 'spearmans_rho'


def convert_to_required_data_types(annotations):
    return [[str(annotator_label), str(item_label), str(tag)] for annotator_label, item_label, tag in annotations.tolist()]


def preprocess_annotations(annotator_one_annotations, annotator_two_annotations):
    annotator_one_id = np.reshape(np.arange(annotator_one_annotations.shape[0]), newshape=(100, 1))
    nltk_required_info_for_annotator_one = np.concatenate((np.zeros((100, 1), dtype=np.int32), annotator_one_id), axis=1)
    annotator_one_annotations = np.concatenate((nltk_required_info_for_annotator_one, np.reshape(annotator_one_annotations, newshape=(100, 1))), axis=1)

    annotator_one_annotations = convert_to_required_data_types(annotator_one_annotations)

    annotator_two_id = np.reshape(np.arange(annotator_two_annotations.shape[0]), newshape=(100, 1))
    nltk_required_info_for_annotator_two = np.concatenate((np.ones((100, 1)), annotator_two_id), axis=1)
    annotator_two_annotations = np.concatenate((nltk_required_info_for_annotator_two, np.reshape(annotator_two_annotations, newshape=(100, 1))), axis=1)

    annotator_two_annotations = convert_to_required_data_types(annotator_two_annotations)

    return np.concatenate((annotator_one_annotations, annotator_two_annotations))


def calculate_cohens_kappa(annotator_one_annotations, annotator_two_annotations):
    rating = agreement.AnnotationTask(data=preprocess_annotations(annotator_one_annotations, annotator_two_annotations))

    return {cohens_kappa_label: rating.kappa(),
            weighted_cohens_kappa_label: rating.weighted_kappa()}


def calculate_agreement_scores(annotator_one_annotations, annotator_two_annotations):
    cohens_kappa = calculate_cohens_kappa(annotator_one_annotations, annotator_two_annotations)

    kendalls_tau = kendalltau(annotator_one_annotations, annotator_two_annotations)
    pearsons_r = pearsonr(annotator_one_annotations, annotator_two_annotations)
    spearmans_rho = spearmanr(annotator_one_annotations, annotator_two_annotations)

    return {**cohens_kappa, **{kendalls_tau_label: kendalls_tau,
                               pearsons_r_label: pearsons_r,
                               spearmans_rho_label: spearmans_rho}}


def indexed_mean(list_of_tuples, index):
    return np.array([tuple[index] for tuple in list_of_tuples]).mean()


def tuple_mean(list_of_tuples):
    return indexed_mean(list_of_tuples, 0), indexed_mean(list_of_tuples, 1)


def calculate_inter_annotator_agreement(annotations):
    cohens_kappa_scores = []
    weighted_cohens_kappa_scores = []
    kendalls_tau_scores = []
    pearson_correlations = []
    spearman_correlations = []

    for annotator_one, annotator_two in combinations([i for i in range(annotations.shape[1])], 2):
        agreement_scores = calculate_agreement_scores(annotations[:, annotator_one], annotations[:, annotator_two])

        cohens_kappa_scores.append(agreement_scores[cohens_kappa_label])
        weighted_cohens_kappa_scores.append(agreement_scores[weighted_cohens_kappa_label])
        kendalls_tau_scores.append(agreement_scores[kendalls_tau_label])
        pearson_correlations.append(agreement_scores[pearsons_r_label])
        spearman_correlations.append(agreement_scores[spearmans_rho_label])

    cohens_kappa_mean = np.array(cohens_kappa_scores).mean()
    weighted_cohens_kappa_mean = np.array(weighted_cohens_kappa_scores).mean()
    kendalls_tau_mean = tuple_mean(kendalls_tau_scores)
    pearson_correlation_mean = tuple_mean(pearson_correlations)
    spearman_correlations_mean = tuple_mean(spearman_correlations)

    data = [(cohens_kappa_mean, 1), (weighted_cohens_kappa_mean, 1), kendalls_tau_mean, pearson_correlation_mean, spearman_correlations_mean]
    return pd.DataFrame(data,
                        index=["cohen's kappa", "weighted cohen's kappa", "kendall's tau", "pearson's r", "spearman's rho"],
                        columns=["correlation", "p"])
