import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from src.featurizers.ldo.lund_dighem_olofsson_featurizer import featurize
from src.utils.save_data import save_as_csv, save_fig


class FeatureSet:
    def __init__(self):
        self.features = None
        self.feature_labels = ['H', 'V', 'E']

    def load(self, path_to_projects):
        self.features = featurize(path_to_projects)

    def describe_features(self, output_path):
        self.output_feature_csv(output_path)
        self.output_feature_plots(output_path)

    def output_feature_csv(self, output_path):
        self.output_feature_descriptive_statistics(output_path)
        self.output_feature_correlations(output_path)

    def output_feature_descriptive_statistics(self, output_path):
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(self.features)
        save_as_csv(
            output_path,
            'features_description.csv',
            np.rot90(np.array([minmax[0], minmax[1], mean, variance, skewness, kurtosis])),
            'Min,Max,Mean,Variance,Skewness,Kurtosis'
        )

    def output_feature_correlations(self, output_path):
        df = pd.DataFrame(index=self.feature_labels, columns=self.feature_labels)
        for x, y in itertools.product([0, 1, 2], repeat=2):
            r = spearmanr(self.features[0:, x], self.features[0:, y])
            df[self.feature_labels[y]][self.feature_labels[x]] = r[0]

        df.to_csv(os.path.join(output_path, 'feature_correlations.csv'))

    def output_feature_plots(self, output_path):
        self.output_feature_scatter_plots(output_path)

    def output_feature_scatter_plots(self, output_path):
        plt.subplots_adjust(hspace=0.4, wspace=0.6)

        i = 1
        for x, y in itertools.permutations([0, 1, 2], 2):
            plt.subplot(2, 3, i)
            plt.scatter(self.features[0:, x], self.features[0:, y])

            plt.xlabel(self.feature_labels[x])
            plt.ylabel(self.feature_labels[y])

            i += 1

        save_fig(output_path, 'feature_scatter_plots.png', plt)
        plt.clf()
