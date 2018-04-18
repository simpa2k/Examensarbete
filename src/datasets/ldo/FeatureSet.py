import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.utils.save_data import save_as_csv, save_fig


class FeatureSet:
    def __init__(self, featurize, feature_labels):
        self.features = None
        self.featurize = featurize
        self.feature_labels = feature_labels

    def load(self, path_to_projects, force_feature_generation):
        self.features = self.featurize(path_to_projects, self.feature_labels, force_feature_generation)

    def describe_features(self, output_path):
        self.output_feature_csv(output_path)
        # self.output_feature_plots(output_path)

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
        for x, y in itertools.permutations(self.feature_labels, 2):
            plt.subplot(2, 3, i)
            plt.scatter(self.features[x], self.features[y])

            plt.xlabel(x)
            plt.ylabel(y)

            i += 1

        save_fig(output_path, 'feature_scatter_plots.png', plt)
        plt.clf()
