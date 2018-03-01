import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.featurizers.ldo.lund_dighem_olofsson_featurizer import featurize
from src.datasets.ldo.inter_annotator_agreement import calculate_inter_annotator_agreement
from src.utils.probability_density_function import probability_density_function_from_samples


class LundDighemOlofssonDataset():

    def __init__(self):
        self.features = None
        self.annotations = None
        self.unprocessed_annotations = None

    def load(self, path_to_projects, path_to_annotations):
        self.features = self.load_features(path_to_projects)
        self.annotations, self.unprocessed_annotations = self.load_annotations(path_to_annotations)

    def load_features(self, path_to_projects):
        return featurize(path_to_projects)

    def load_annotations(self, path_to_annotations):
        unprocessed_annotations = np.genfromtxt(path_to_annotations, delimiter=',')[1:, 1:4]
        return unprocessed_annotations.mean(axis=1), unprocessed_annotations

    def describe(self, output_path):
        self.describe_annotations(os.path.join(output_path, 'annotations'))
        self.describe_features(os.path.join(output_path, 'features'))
        self.correlate(os.path.join(output_path, 'correlations'))

    def describe_annotations(self, output_path):
        self.output_annotation_csv(output_path)
        self.output_normal_test(output_path)
        calculate_inter_annotator_agreement(self.unprocessed_annotations)
        self.output_annotation_plots(output_path)

    def output_annotation_csv(self, output_path):
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(self.annotations)
        np.savetxt(os.path.join(output_path, 'annotations_description.csv'),
                   np.array([nobs, minmax[0], minmax[1], mean, variance, skewness, kurtosis])[np.newaxis],
                   header='Number of observations,Min,Max,Mean,Variance,Skewness,Kurtosis',
                   comments='',
                   delimiter=',',
                   fmt='%5.2f')

    def output_normal_test(self, output_path):
        k2, p = stats.normaltest(self.annotations)
        np.savetxt(os.path.join(output_path, 'annotations_normal_test.csv'),
                   np.array([k2, p])[np.newaxis],
                   header='k2,p',
                   comments='',
                   delimiter=',',
                   fmt='%5.2f')

    def output_annotation_plots(self, output_path):
        plt.hist(self.annotations, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        plt.savefig(os.path.join(output_path, 'annotations_histogram.png'))
        plt.clf()

        probability_density_function_from_samples(self.annotations,
                                                  1.5, 5,
                                                  .22,
                                                  os.path.join(output_path, 'annotations_pdf.png'))

    def describe_features(self, output_path):
        self.output_feature_csv(output_path)
        self.output_feature_plots(output_path)

    def output_feature_csv(self, output_path):
        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(self.features)
        np.savetxt(os.path.join(output_path, 'features_description.csv'),
                   np.array([minmax[0], minmax[1], mean, variance, skewness, kurtosis]),
                   header='Min,Max,Mean,Variance,Skewness,Kurtosis',
                   comments='',
                   delimiter=',',
                   fmt='%5.2f')

    def output_feature_plots(self, output_path):
        self.output_feature_scatter_plots(output_path)

    def output_feature_scatter_plots(self, output_path):
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        i = 1
        for x, y in itertools.permutations([0, 1, 2], 2):
            plt.subplot(2, 3, i)
            plt.scatter(self.features[0:, x], self.features[0:, y])
            i += 1

        plt.savefig(os.path.join(output_path, 'feature_scatter_plots.png'))
        plt.clf()

    def correlate(self, output_path):
        pass

