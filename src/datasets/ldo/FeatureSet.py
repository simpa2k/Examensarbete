import os
import itertools

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from src.featurizers.ldo.lund_dighem_olofsson_featurizer import featurize, feature_labels
from src.utils.save_data import save_as_csv, save_fig


class FeatureSet:
    def __init__(self):
        self.features = None

    def load(self, path_to_projects):
        self.features = featurize(path_to_projects)

    def describe_features(self, output_path):
        self.output_feature_csv(output_path)
        self.output_feature_plots(output_path)

    def output_feature_csv(self, output_path):
        self.output_feature_descriptive_statistics(output_path)

    def output_feature_descriptive_statistics(self, output_path):
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(self.features.as_matrix())
        save_as_csv(
            output_path,
            'features_description.csv',
            np.rot90(np.array([minmax[0], minmax[1], mean, variance, skewness, kurtosis])),
            'Minimum,Maximum,Medelvärde,Varians,Skevhet,Kurtosis'
        )

    def output_feature_plots(self, output_path):
        self.output_feature_scatter_plots(output_path)

    def output_feature_scatter_plots(self, output_path):
        plt.subplots_adjust(hspace=0.4, wspace=0.6)

        i = 1
        for x, y in itertools.permutations(feature_labels, 2):
            plt.subplot(2, 3, i)
            plt.scatter(self.features[x], self.features[y])

            plt.xlabel(x)
            plt.ylabel(y)

            i += 1

        save_fig(output_path, 'feature_scatter_plots.png', plt)
        plt.clf()
