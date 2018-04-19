import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.datasets.ldo.FeatureSet import FeatureSet
from src.datasets.ldo.AnnotationSet import AnnotationSet

from src.featurizers.ldo.lund_dighem_olofsson_featurizer import get_featurizer, featurize_project_with_project_level_features, featurize_project_with_mean_method_level_features, featurize_project_with_all_features
from src.utils.save_data import save_fig


class LundDighemOlofssonDataset():
    def __init__(self):
        feature_labels = ['Rader kod', 'Halsteads V', 'Entropi']
        all_feature_labels = ['Rader kod, projekt', 'Halsteads V, projekt', 'Entropi, projekt',
                              'Rader kod, medelvärde metod', 'Halsteads V, medelvärde metod']

        self.project_level_feature_set = FeatureSet(
            get_featurizer(featurize_project_with_project_level_features, 'project_level_features.csv'),
            feature_labels
        )
        self.mean_method_level_feature_set = FeatureSet(
            get_featurizer(featurize_project_with_mean_method_level_features, 'mean_method_level_features.csv'),
            feature_labels
        )
        self.complete_feature_set = FeatureSet(
            get_featurizer(featurize_project_with_all_features, 'all_features.csv'),
            all_feature_labels
        )

        self.annotation_set = AnnotationSet()
        self.data = None

        plt.style.use('seaborn')

    def load(self, path_to_projects, path_to_annotations, force_feature_generation):
        self.project_level_feature_set.load(path_to_projects, force_feature_generation)
        self.mean_method_level_feature_set.load(path_to_projects, force_feature_generation)
        self.complete_feature_set.load(path_to_projects, force_feature_generation)

        self.annotation_set.load_annotations(path_to_annotations)

        self.project_level_feature_data = self.annotation_set.binarized_annotations.join(self.project_level_feature_set.features)
        self.mean_method_level_feature_data = self.annotation_set.binarized_annotations.join(self.mean_method_level_feature_set.features)
        self.complete_feature_data = self.annotation_set.binarized_annotations.join(self.complete_feature_set.features)

    def get_project_level_features(self):
        return self.project_level_feature_data[self.project_level_feature_set.feature_labels].as_matrix()

    def get_mean_method_level_features(self):
        return self.mean_method_level_feature_data[self.mean_method_level_feature_set.feature_labels].as_matrix()

    def get_all_features(self):
        return self.complete_feature_data[self.complete_feature_set.feature_labels].as_matrix()

    def get_annotations(self):
        return self.project_level_feature_data[self.annotation_set.annotation_column].as_matrix()

    def describe(self, output_path):
        self.project_level_feature_set.describe_features(os.path.join(output_path, 'features/project_level_features'))
        self.mean_method_level_feature_set.describe_features(os.path.join(output_path, 'features/mean_method_level_features'))
        self.complete_feature_set.describe_features(os.path.join(output_path, 'features/all_features'))

        self.annotation_set.describe_annotations(os.path.join(output_path, 'annotations'))

        self.correlate_features_and_votes(self.project_level_feature_set, os.path.join(output_path, 'features/project_level_features'))
        self.correlate_features_and_votes(self.mean_method_level_feature_set, os.path.join(output_path, 'features/mean_method_level_features'))
        self.correlate_features_and_votes(self.complete_feature_set, os.path.join(output_path, 'features/all_features'))

        self.output_data(self.project_level_feature_set, os.path.join(output_path, 'annotations_and_project_level_features.csv'))
        self.output_data(self.mean_method_level_feature_set, os.path.join(output_path, 'annotations_and_mean_method_level_features.csv'))
        self.output_data(self.complete_feature_set, os.path.join(output_path, 'annotations_and_all_features.csv'))

    def correlate_features_and_votes(self, feature_set, output_path):
        labels = np.concatenate((feature_set.feature_labels, ['Bedömning']))
        concatenated = np.concatenate(
            (feature_set.features, np.reshape(self.annotation_set.averaged_annotations, newshape=(100, 1))),
            axis=1)

        correlations = pd.DataFrame(index=labels, columns=labels, dtype=float)
        p_values = pd.DataFrame(index=labels, columns=labels, dtype=float)

        for x, y in itertools.product(range(len(labels)), repeat=2):
            r = spearmanr(concatenated[0:, x], concatenated[0:, y])

            correlations[labels[y]][labels[x]] = r[0]
            p_values[labels[y]][labels[x]] = r[1]

        correlations.round(2).to_csv(os.path.join(output_path, 'correlations.csv'), index_label='x')
        p_values.round(4).to_csv(os.path.join(output_path, 'correlation_p_values.csv'), index_label='x')

    def output_data(self, feature_set, output_path):
        to_modify = pd.DataFrame(self.annotation_set.averaged_annotations, columns=['Bedömning']).join(feature_set.features)
        to_modify.index.name = 'Uppgift'

        to_modify.to_csv(output_path)

