import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.datasets.ldo.FeatureSet import FeatureSet
from src.datasets.ldo.AnnotationSet import AnnotationSet

from src.featurizers.ldo.lund_dighem_olofsson_featurizer import feature_labels


class LundDighemOlofssonDataset():
    def __init__(self):
        self.feature_set = FeatureSet()
        self.annotation_set = AnnotationSet()
        self.data = None

        plt.style.use('seaborn')

    def load(self, path_to_projects, path_to_annotations):
        self.feature_set.load(path_to_projects)
        self.annotation_set.load_annotations(path_to_annotations)

        self.data = self.annotation_set.binarized_annotations.join(self.feature_set.features)

    def get_features(self):
        return self.data[feature_labels].as_matrix()

    def get_annotations(self):
        return self.data[self.annotation_set.annotation_column].as_matrix()

    def describe(self, output_path):
        self.feature_set.describe_features(os.path.join(output_path, 'features'))
        self.annotation_set.describe_annotations(os.path.join(output_path, 'annotations'))
        self.correlate_features_and_votes(output_path)

    def correlate_features_and_votes(self, output_path):
        labels = np.concatenate((feature_labels, ['Bedömning']))
        concatenated = np.concatenate(
            (self.feature_set.features, np.reshape(self.annotation_set.averaged_annotations, newshape=(100, 1))),
            axis=1)
        df = pd.DataFrame(index=labels, columns=labels, dtype=float)

        for x, y in itertools.product(range(len(labels)), repeat=2):
            r = spearmanr(concatenated[0:, x], concatenated[0:, y])
            df[labels[y]][labels[x]] = r[0]

        df.round(2).to_csv(os.path.join(output_path, 'correlations.csv'))
